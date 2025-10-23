"""
rank.py
~~~~~~~~

This module implements a simplified yet fully featured ranking engine for
evaluating pairs of passages on a set of attributes.  It draws heavy
inspiration from the existing ``elo.py`` implementation found in the
GABRIEL distribution but removes support for the classic Elo rating
system and focuses solely on the Bradley–Terry (BT) style approach.

Key improvements and changes relative to ``elo.py`` include:

* A streamlined configuration dataclass (`RankConfig`) that exposes the
  parameters most relevant to the BT method.  Irrelevant options
  (e.g. ``rating_method``, ``k_factor``) have been removed, and
  parameter names have been harmonised with the high‑level API
  described in the calling code.  ``file_name`` is now treated as a
  stem; if an extension is provided it will be stripped automatically.

* Support for the new rankings prompt (``rankings_prompt.jinja2``)
  which allows the large language model to return one of four
  outcomes for each attribute: ``"circle"``, ``"square"``, ``"draw``
  or ``"insufficient signal"``.  ``draw`` and ``insufficient signal``
  are both treated as a tie and contribute equally to both items when
  fitting the BT model.

* A cleaned up asynchronous ``run`` method that accepts a pandas
  ``DataFrame`` and the name of the column containing the text to be
  ranked.  Each row receives a stable identifier derived from a hash of its
  contents; no external ``id_col`` argument is required.  The method
  produces a DataFrame with one row per input passage, a numeric
  rating for each attribute, optional z‑scores and standard errors,
  and writes the results to disk under ``save_dir``.

The core ranking logic remains largely unchanged from ``elo.py``
because the underlying mathematics of the BT model and the pairing
strategies continue to work well.  However, comments have been added
throughout the code to clarify intent and to highlight areas where
further experimentation (e.g. alternative information gain metrics) can
be incorporated.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import hashlib
import math
import copy
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Sequence
import json

import numpy as np
import pandas as pd

# Import helper utilities from the gabriel package.  These modules are
# expected to be available in the runtime environment.  Should you wish
# to run this module outside of the GABRIEL distribution, you may need
# to adjust these imports accordingly.
from gabriel.core.prompt_template import PromptTemplate, resolve_template
from gabriel.utils.openai_utils import get_all_responses
from gabriel.utils import (
    safest_json,
    load_image_inputs,
    load_audio_inputs,
)
from .rate import Rate, RateConfig


@dataclass
class RankConfig:
    """User‑visible configuration for :class:`Rank`.

    Only a minimal set of parameters are exposed to keep the API
    straightforward.  Additional hyperparameters for the underlying
    Bradley–Terry model and pairing heuristics are fixed at sensible
    values and should not generally need to be changed.  See the
    surrounding documentation for more details.

    Parameters
    ----------
    attributes:
        Mapping from attribute names to definitions.  A list of
        attribute names is also accepted; definitions will be set to
        empty strings.
    n_rounds:
        Number of rounds of pairwise comparisons to perform.
    matches_per_round:
        Number of matches per item per round.
    power_matching:
        Whether to use an information‑theoretic pairing heuristic.
    add_zscore:
        If ``True`` the per-attribute columns in the output DataFrame contain
        z‑scores (normalised scores).  The raw Bradley–Terry log‑skill
        estimates are always written alongside these z‑scores using an
        ``"<attribute>_raw"`` suffix.
    compute_se:
        If ``True`` compute standard errors for each score and include
        ``"<attribute>_se"`` columns in the output.  Standard errors are
        primarily useful for diagnosing the pairing strategy and are disabled
        by default so that only the renamed z‑score columns are returned.
    learning_rate:
        Pseudo‑count used by the BT model to regularise the win/loss
        matrix.  A larger value makes updates more conservative.
    model:
        Name of the language model to call via ``get_all_responses``.
    n_parallels:
        Number of parallel API calls to issue.
    use_dummy:
        Whether to use a dummy model for testing purposes.
    save_dir:
        Directory into which result files should be saved.
    file_name:
        Stem for the output CSV files.  If an extension is present it
        will be removed.
    additional_instructions:
        Extra, user‑supplied instructions passed to the prompt.
    recursive:
        When ``True`` run ranking in multiple stages, pruning the pool
        of candidates between stages according to ``recursive_fraction``
        and ``recursive_min_remaining``.
    recursive_fraction, recursive_min_remaining,
    recursive_final_round_multiplier:
        Parameters controlling how many items are kept between stages
        and how many rounds are executed in the final stage when
        ``recursive`` is enabled.
    recursive_cut_attr, recursive_cut_side:
        Select which attribute and direction are used when choosing
        which items survive to the next stage.
    recursive_rate_first_round:
        If ``True`` perform a :class:`Rate` sweep before the first
        recursive stage and seed subsequent rounds with those scores.
    recursive_rewrite_func, recursive_rewrite_text_col:
        Optional hook to rewrite surviving passages between stages and
        the column where rewritten text should be stored.
    recursive_keep_stage_columns, recursive_add_stage_suffix:
        Control whether intermediate stage outputs are merged into the
        final results and whether their columns receive stage prefixes.
    max_timeout:
        Optional upper bound for individual API calls when retrieving
        ranking judgements. ``None`` (default) lets the timeout be
        derived dynamically from observed latencies in
        :func:`gabriel.utils.openai_utils.get_all_responses`.
    initial_rating_pass:
        Enables a one-off :class:`Rate` pass before standard ranking
        rounds.  The centred scores from that pass seed the initial
        Bradley–Terry ratings which helps pairing focus on refinement.
    rate_kwargs:
        Optional dictionary of overrides forwarded to the rating task
        whenever it is invoked (either as a seed or during recursion).
    """

    attributes: Union[Dict[str, str], List[str]]
    n_rounds: int = 5
    matches_per_round: int = 5
    power_matching: bool = True
    add_zscore: bool = True
    compute_se: bool = False
    learning_rate: float = 0.1
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    use_dummy: bool = False
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "rankings"
    additional_instructions: Optional[str] = None
    circle_first: Optional[bool] = None
    modality: str = "text"
    n_attributes_per_run: int = 8
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    max_timeout: Optional[float] = None
    # Recursive execution controls
    recursive: bool = False
    recursive_fraction: float = 1.0 / 3.0
    recursive_min_remaining: int = 30
    recursive_final_round_multiplier: int = 3
    recursive_cut_attr: Optional[str] = None
    recursive_cut_side: str = "top"
    recursive_rate_first_round: bool = False
    recursive_rewrite_func: Optional[Callable[[str, str, int], str]] = None
    recursive_rewrite_text_col: str = "text"
    recursive_keep_stage_columns: bool = True
    recursive_add_stage_suffix: bool = True
    # Optional single pass rating seed controls
    initial_rating_pass: bool = False
    rate_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Rank:
    """Rank items by comparing passages pairwise on multiple attributes.

    An instance of :class:`Ranker` orchestrates the iterative process
    of sampling pairs, calling a language model to adjudicate which
    passage better exhibits each attribute, and then fitting a
    Bradley–Terry model to those outcomes.  Standard errors and
    z‑scores are optionally computed.  Results are persisted to disk
    after the final round.
    """

    def __init__(
        self,
        cfg: RankConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        """Instantiate a ranking engine.

        Parameters
        ----------
        cfg:
            User‑provided configuration.
        template:
            Optional :class:`gabriel.core.prompt_template.PromptTemplate` to
            render the comparison prompts.  If not supplied, the built‑in
            ``rankings_prompt.jinja2`` template is used.
        template_path:
            Path to a custom prompt template on disk. The template is
            validated to ensure it expects the same variables as the
            built‑in template.
        """
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="rankings_prompt.jinja2",
        )
        # random state; a seed is intentionally omitted from the public
        # configuration to discourage brittle behaviour.  If
        # reproducibility is required, modify this line to pass a
        # specific seed.
        self.rng = random.Random()
        # place holders for multiway rankings and aggregated standard errors
        self.history_multi: Dict[str, List[List[str]]] = {}
        self._last_se_agg: Optional[Dict[str, float]] = None

        # internal constants for the pairing and BT algorithms.  These
        # values are deliberately not exposed through the public API as
        # they seldom need tuning and adjusting them can complicate
        # reproducibility.  Should you need to experiment with these
        # hyperparameters, modify the values below.
        self._EXPLORE_FRAC = 0.2  # fraction of random pairings per round
        self._CANDIDATE_NEIGHBORS = 20  # neighbourhood size for info gain pairing
        self._HIGH_SE_FRAC = 0.25  # fraction of high‑uncertainty items
        self._MAX_ITER = 1000  # maximum iterations for BT optimisation
        self._TOL = 1e-6  # convergence tolerance for BT
        # A small ridge term stabilises the inversion of the Fisher information
        # matrix when computing standard errors.  The previous value (1e‑9)
        # occasionally led to extremely large uncertainties for items with
        # limited or contradictory comparisons.  Increasing this value
        # regularises the covariance estimate and prevents unreasonably
        # large standard errors.  If you observe inflated SE values,
        # consider increasing this further (e.g. to 1e‑4).
        self._SE_RIDGE = 1e-5

        # The maximum number of candidate pairs to consider per pairing round.
        # When the number of items becomes very large (e.g. tens of thousands),
        # evaluating all possible pairs is intractable.  We therefore cap the
        # total number of candidate pairs by limiting the neighbourhood size
        # used when constructing candidate pairs.  The default of 200k ensures
        # that information gain pairing remains tractable even with very
        # large data sets: for example, with 10 000 items and a cap of
        # 200 000, each item will only consider approximately 20 neighbours.
        self._MAX_CANDIDATE_PAIRS_PER_ROUND = 200_000

        # ------------------------------------------------------------------
        # Public API for adding multiway rankings
        # ------------------------------------------------------------------
    def add_multiway_ranking(self, attr: str, ranking: List[str]) -> None:
        """Record a multiway ranking for a given attribute.

        Multiway rankings are stored but not used by the current BT
        implementation.  They are retained for potential future
        extensions where a Plackett–Luce model could be incorporated.
        """
        if attr not in self.history_multi:
            self.history_multi[attr] = []
        self.history_multi[attr].append(ranking)

    def _attributes_as_dict(self) -> Dict[str, str]:
        if isinstance(self.cfg.attributes, dict):
            return dict(self.cfg.attributes)
        return {attr: "" for attr in self.cfg.attributes}

    def _split_rate_kwargs(self, overrides: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        merged: Dict[str, Any] = {}
        if self.cfg.rate_kwargs:
            merged.update(self.cfg.rate_kwargs)
        if overrides:
            merged.update(overrides)
        config_fields = {f.name for f in fields(RateConfig)}
        cfg_kwargs: Dict[str, Any] = {}
        run_kwargs: Dict[str, Any] = {}
        for key, value in merged.items():
            if key in config_fields:
                cfg_kwargs[key] = value
            else:
                run_kwargs[key] = value
        return cfg_kwargs, run_kwargs

    async def _run_rate_pass(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        save_dir: str,
        file_name: str,
        reset_files: bool,
        rate_kwargs: Optional[Dict[str, Any]] = None,
        runtime_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        cfg_overrides, run_kwargs = self._split_rate_kwargs(rate_kwargs)
        rate_cfg = RateConfig(
            attributes=self._attributes_as_dict(),
            save_dir=save_dir,
            file_name=file_name,
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            additional_instructions=self.cfg.additional_instructions or "",
            modality=self.cfg.modality,
            n_attributes_per_run=self.cfg.n_attributes_per_run,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            max_timeout=self.cfg.max_timeout,
        )
        for key, value in cfg_overrides.items():
            setattr(rate_cfg, key, value)
        combined_kwargs = dict(run_kwargs)
        if runtime_kwargs:
            combined_kwargs.update(runtime_kwargs)
        combined_kwargs.setdefault("web_search", self.cfg.modality == "web")
        rate_task = Rate(rate_cfg)
        return await rate_task.run(
            df,
            column_name,
            reset_files=reset_files,
            **combined_kwargs,
        )

    def _seed_ratings_from_rate(
        self,
        rate_df: pd.DataFrame,
        *,
        id_column: Optional[str],
        text_column: str,
        item_ids: Sequence[str],
        attr_keys: Sequence[str],
    ) -> Dict[str, Dict[str, float]]:
        if rate_df.empty:
            return {}
        attr_cols = [attr for attr in attr_keys if attr in rate_df.columns]
        if not attr_cols:
            return {}
        if id_column and id_column in rate_df.columns:
            key_series = rate_df[id_column].astype(str)
        elif text_column in rate_df.columns:
            key_series = rate_df[text_column].astype(str).map(
                lambda x: hashlib.sha1(x.encode()).hexdigest()[:8]
            )
        else:
            return {}
        stage_df = pd.DataFrame({"_id": key_series})
        for attr in attr_cols:
            stage_df[attr] = pd.to_numeric(rate_df[attr], errors="coerce")
        grouped = stage_df.groupby("_id")[attr_cols].mean()
        seeds: Dict[str, Dict[str, float]] = {}
        for attr in attr_cols:
            series = grouped[attr].dropna()
            if series.empty:
                continue
            mean_val = float(series.mean())
            centred = series - mean_val
            for item_id, value in centred.items():
                seeds.setdefault(item_id, {})[attr] = float(value)
        # Only retain seeds for items that will appear in the ranking loop
        return {item_id: seeds[item_id] for item_id in item_ids if item_id in seeds}

    # ------------------------------------------------------------------
    # BT / PL fitting utilities
    # ------------------------------------------------------------------
    def _fit_bt(
        self,
        item_ids: List[str],
        outcomes: List[Tuple[str, str]],
        pseudo: float,
        max_iter: int,
        tol: float,
        return_info: bool = False,
    ) -> Union[Dict[str, float], Tuple[Dict[str, float], np.ndarray, np.ndarray]]:
        """Fit a Bradley–Terry model given pairwise outcomes.

        Parameters
        ----------
        item_ids:
            List of unique item identifiers.
        outcomes:
            List of tuples ``(winner, loser)`` representing outcomes of
            pairwise matches.  Ties can be represented by including
            both ``(a, b)`` and ``(b, a)`` in the list; each entry
            contributes a single increment to the win matrix.
        pseudo:
            Pseudo count added to both win and total match counts.  Acts
            as a smoothing prior.
        max_iter, tol:
            Control convergence of the iterative fixed‑point updates.
        return_info:
            If ``True`` return the intermediate matrices ``n_ij`` and
            ``p_ij`` for downstream standard error computation.

        Returns
        -------
        scores : dict
            Mapping from item identifier to estimated log‑skill.
        (scores, n_ij, p_ij) : tuple
            When ``return_info`` is ``True``, also return the total
            match counts and predicted win probabilities for each pair.
        """
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        # win matrix; wins[i,j] counts how many times i beat j
        wins = np.zeros((n, n), dtype=float)
        for w, l in outcomes:
            if w in idx and l in idx:
                wins[idx[w], idx[l]] += 1.0
        # total matches between each pair
        n_ij = wins + wins.T
        # total wins for each item
        w_i = wins.sum(axis=1)
        # add pseudo counts
        n_ij += pseudo
        w_i += pseudo
        # initialise skill parameters uniformly
        p = np.ones(n, dtype=float)
        for _ in range(max_iter):
            # denominator for each player in the fixed point update
            denom = (n_ij / (p[:, None] + p[None, :])).sum(axis=1)
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new
        # convert to log space and centre at zero mean
        s = np.log(p)
        s -= s.mean()
        if not return_info:
            return {item: float(val) for item, val in zip(item_ids, s)}
        # predicted win probabilities between each pair
        exp_s = np.exp(s)
        p_ij = exp_s[:, None] / (exp_s[:, None] + exp_s[None, :])
        return {item: float(val) for item, val in zip(item_ids, s)}, n_ij, p_ij

    def _bt_standard_errors(
        self,
        s: np.ndarray,
        n_ij: np.ndarray,
        p_ij: np.ndarray,
        ridge: float,
    ) -> np.ndarray:
        """Estimate standard errors for BT skill parameters.

        The observed Fisher information for the Bradley–Terry model is given by
        ``I = diag(q 1) - q`` where ``q = n_ij * p_ij * (1 - p_ij)`` encodes the
        uncertainty contributed by each pairwise comparison (Ford, 1957).  The
        estimates satisfy a sum-to-zero constraint, so the Fisher information is
        rank deficient with the all-ones vector in its null space.  Instead of
        selecting an arbitrary reference item (which previously produced
        inflated standard errors for that reference when it received few
        comparisons), we project the matrix onto the constrained subspace and
        take its Moore–Penrose pseudoinverse.  A small ridge term stabilises the
        inversion for sparse comparison graphs.  The standard error for item ``i``
        is the square root of the ``i``-th diagonal entry of the resulting
        covariance matrix.

        Parameters
        ----------
        s : np.ndarray
            Array of estimated log-skills for each item.
        n_ij : np.ndarray
            Matrix of total match counts between items (wins + losses).
        p_ij : np.ndarray
            Matrix of predicted win probabilities between items.
        ridge : float
            Small constant added to the diagonal of the projected Fisher
            information matrix for numerical stability.

        Returns
        -------
        np.ndarray
            Array of standard errors corresponding to each element of ``s``.
        """

        n = len(s)
        if n == 0:
            return np.array([], dtype=float)
        if n == 1:
            return np.zeros(1, dtype=float)

        q_ij = n_ij * p_ij * (1 - p_ij)
        diag = q_ij.sum(axis=1)
        I = np.diag(diag) - q_ij
        I = np.nan_to_num(I)
        ones = np.ones((n, 1))
        proj = np.eye(n) - ones @ ones.T / n
        I_proj = proj @ I @ proj
        I_proj[np.diag_indices(n)] += ridge
        try:
            cov = np.linalg.pinv(I_proj, rcond=1e-12)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(np.nan_to_num(I_proj), rcond=1e-12)
        cov = proj @ cov @ proj
        cov = 0.5 * (cov + cov.T)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return np.nan_to_num(se)

    def _fit_pl(
        self,
        item_ids: List[str],
        rankings: List[List[str]],
        pseudo: float,
        max_iter: int,
        tol: float,
    ) -> Dict[str, float]:
        """Fit a Plackett–Luce model for multiway rankings.

        When every ranking is of length two this reduces to the BT
        model and defers to :meth:`_fit_bt`.  If no rankings are
        provided a zero‑centred score is returned for each item.  See
        Hunter (2004) for details on the fitting procedure.
        """
        if not rankings:
            return {i: 0.0 for i in item_ids}
        # if all rankings are of length 2, delegate to BT
        if all(len(r) == 2 for r in rankings):
            outcomes = [(r[0], r[1]) for r in rankings]
            return self._fit_bt(
                item_ids, outcomes, pseudo, max_iter, tol, return_info=False
            )
        n = len(item_ids)
        idx = {item: i for i, item in enumerate(item_ids)}
        w_i = np.zeros(n, dtype=float)
        rankings_idx = []
        for r in rankings:
            r_idx = [idx[x] for x in r if x in idx]
            if len(r_idx) < 2:
                continue
            rankings_idx.append(r_idx)
            for i_ in r_idx:
                w_i[i_] += 1.0
        if len(rankings_idx) == 0:
            return {i: 0.0 for i in item_ids}
        w_i += pseudo
        p = np.ones(n, dtype=float)
        for _ in range(max_iter):
            denom = np.zeros(n, dtype=float)
            for r_idx in rankings_idx:
                remaining = np.array(r_idx, dtype=int)
                sum_p = p[remaining].sum()
                for i_ in r_idx:
                    denom[i_] += 1.0 / sum_p
                    sum_p -= p[i_]
            denom[denom == 0] = 1e-12
            p_new = w_i / denom
            if np.max(np.abs(p_new - p)) < tol:
                p = p_new
                break
            p = p_new
        s = np.log(p)
        s -= s.mean()
        return {item: float(val) for item, val in zip(item_ids, s)}

    # ------------------------------------------------------------------
    # Pairing strategies
    # ------------------------------------------------------------------
    def _pairs_random(
        self, item_ids: List[str], texts_by_id: Dict[str, str], mpr: int
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Return a set of random, unique pairs for the given items."""
        pairs_set: set[Tuple[str, str]] = set()
        for a in item_ids:
            others = [x for x in item_ids if x != a]
            if not others:
                continue
            k = min(mpr, len(others))
            opponents = self.rng.sample(others, k)
            for b in opponents:
                pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_adjacent(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Pair each item with its nearest neighbours in rating space."""
        pairs_set: set[Tuple[str, str]] = set()
        sorted_ids = sorted(item_ids, key=lambda i: current_ratings[i])
        n = len(sorted_ids)
        for i, a in enumerate(sorted_ids):
            for off in range(1, mpr + 1):
                b = sorted_ids[(i + off) % n]
                if a == b:
                    continue
                pairs_set.add(tuple(sorted((a, b))))
        # small amount of random exploration to avoid pathological pairings
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        for _ in range(n_random_targets):
            if n < 2:
                break
            a, b = self.rng.sample(item_ids, 2)
            pairs_set.add(tuple(sorted((a, b))))
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_set]

    def _pairs_info_gain(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Dict[str, float],
        se_agg: Dict[str, float],
        mpr: int,
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Select pairs by maximising expected information gain while ensuring
        that every item participates in the prescribed number of matches.

        This implementation differs from the original heuristics by
        considering a bounded set of candidate pairs that scales with the
        number of items.  Each pair is assigned a score based on the
        expected reduction in uncertainty (estimated from the current
        ratings and aggregated standard errors).  Pairs with larger
        scores are chosen first, subject to the constraint that each
        item is matched exactly ``mpr`` times.  If some items remain
        unmatched after exhausting the scored pairs, additional pairs
        are filled in randomly to satisfy the per‑item quota.
        """
        n = len(item_ids)
        if n < 2:
            return []
        max_pairs = max(1, self._MAX_CANDIDATE_PAIRS_PER_ROUND)
        desired_neighbors = max_pairs // max(1, n)
        candidate_neighbors = max(
            mpr, min(self._CANDIDATE_NEIGHBORS, desired_neighbors)
        )

        def logistic_clip(x: float) -> float:
            if x > 50:
                return 1.0
            if x < -50:
                return 0.0
            return 1.0 / (1.0 + np.exp(-x))

        ids_sorted = sorted(item_ids, key=lambda i: current_ratings[i])
        idx_of = {i_id: k for k, i_id in enumerate(ids_sorted)}
        num_high_se = max(1, int(self._HIGH_SE_FRAC * n))
        high_se_ids = sorted(item_ids, key=lambda i: se_agg.get(i, 1.0), reverse=True)[
            :num_high_se
        ]
        candidate_pairs_set: set[Tuple[str, str]] = set()
        for i_id in item_ids:
            pos = idx_of[i_id]
            lower = max(0, pos - candidate_neighbors)
            upper = min(n, pos + candidate_neighbors + 1)
            for j in ids_sorted[lower:upper]:
                if i_id == j:
                    continue
                candidate_pairs_set.add(tuple(sorted((i_id, j))))
        for hs in high_se_ids:
            others = [x for x in item_ids if x != hs]
            k = min(candidate_neighbors, len(others))
            samp = self.rng.sample(others, k)
            for j in samp:
                candidate_pairs_set.add(tuple(sorted((hs, j))))
        remaining_capacity = max_pairs - len(candidate_pairs_set)
        n_random_targets = int(self._EXPLORE_FRAC * n * mpr)
        if remaining_capacity > 0:
            n_random_targets = min(n_random_targets, remaining_capacity)
            for _ in range(n_random_targets):
                if n < 2:
                    break
                a, b = self.rng.sample(item_ids, 2)
                candidate_pairs_set.add(tuple(sorted((a, b))))
        partners_count = {i: 0 for i in item_ids}
        for a, b in candidate_pairs_set:
            partners_count[a] += 1
            partners_count[b] += 1
        for i_id in item_ids:
            while partners_count[i_id] < mpr:
                potential = [x for x in item_ids if x != i_id]
                if not potential:
                    break
                j = self.rng.choice(potential)
                pair = tuple(sorted((i_id, j)))
                if pair not in candidate_pairs_set:
                    candidate_pairs_set.add(pair)
                    partners_count[i_id] += 1
                    partners_count[j] += 1
                else:
                    partners_count[i_id] += 1
                    partners_count[j] += 1
        scored_pairs: List[Tuple[float, str, str]] = []
        for a, b in candidate_pairs_set:
            diff = current_ratings[a] - current_ratings[b]
            p = logistic_clip(diff)
            outcome_var = p * (1 - p)
            var_a = se_agg.get(a, 1.0) ** 2
            var_b = se_agg.get(b, 1.0) ** 2
            param_unc = var_a + var_b
            score = outcome_var * param_unc
            scored_pairs.append((score, a, b))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        needed: Dict[str, int] = {i: mpr for i in item_ids}
        pairs_selected: List[Tuple[str, str]] = []
        pairs_seen: set[Tuple[str, str]] = set()
        for score, a, b in scored_pairs:
            if needed[a] > 0 and needed[b] > 0:
                tup = (a, b) if a < b else (b, a)
                if tup in pairs_seen:
                    continue
                pairs_selected.append(tup)
                pairs_seen.add(tup)
                needed[a] -= 1
                needed[b] -= 1
        while any(cnt > 0 for cnt in needed.values()):
            ids_needing = [i for i, cnt in needed.items() if cnt > 0]
            if not ids_needing:
                break
            # Choose an item that still needs matches
            a = self.rng.choice(ids_needing)
            # Try to pair it with any other item (not just those needing matches) to avoid self‑pairs
            potential = [x for x in item_ids if x != a]
            if not potential:
                # Degenerate case: only one item exists; cannot form a valid pair
                break
            b = self.rng.choice(potential)
            tup = (a, b) if a < b else (b, a)
            pairs_selected.append(tup)
            needed[a] -= 1
            needed[b] -= 1
        return [((a, texts_by_id[a]), (b, texts_by_id[b])) for a, b in pairs_selected]

    def _generate_pairs(
        self,
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        current_ratings: Optional[Dict[str, float]],
        se_agg: Optional[Dict[str, float]],
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """Dispatch to the appropriate pairing strategy."""
        mpr = max(1, self.cfg.matches_per_round)
        # Always use information gain pairing to guarantee exact match counts
        if current_ratings is None:
            current_ratings = {i: 0.0 for i in item_ids}
        if se_agg is None or len(se_agg) != len(item_ids):
            se_full = {i: 1.0 for i in item_ids}
        else:
            se_full = se_agg
        return self._pairs_info_gain(
            item_ids, texts_by_id, current_ratings, se_full, mpr
        )

    async def _catch_up_existing_rounds(
        self,
        new_ids: List[str],
        round_indices: List[int],
        item_ids: List[str],
        texts_by_id: Dict[str, str],
        images_by_id: Dict[str, List[str]],
        audio_by_id: Dict[str, List[Dict[str, str]]],
        attr_batches: List[List[str]],
        attr_keys: List[str],
        history_pairs: Dict[str, List[Tuple[str, str]]],
        ratings: Dict[str, Dict[str, float]],
        se_store: Dict[str, Dict[str, float]],
        base_name: str,
        df_proc: pd.DataFrame,
        _write_checkpoint: Callable[[], None],
        current_ratings: Optional[Dict[str, float]],
        se_agg_local: Optional[Dict[str, float]],
        reset_files: bool,
        **kwargs: Any,
    ) -> None:
        if not new_ids:
            return
        for rnd in round_indices:
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            if not os.path.exists(round_path):
                continue
            try:
                df_round = pd.read_csv(round_path)
            except Exception:
                continue
            counts: Dict[str, int] = {}
            if {"IdA", "IdB"}.issubset(df_round.columns):
                for a, b in zip(df_round["IdA"], df_round["IdB"]):
                    counts[str(a)] = counts.get(str(a), 0) + 1
                    counts[str(b)] = counts.get(str(b), 0) + 1
            else:
                for ident in df_round.get("Identifier", []):
                    parts = str(ident).split("|")
                    if len(parts) != 5:
                        continue
                    _, _, _, id_a, id_b = parts
                    counts[id_a] = counts.get(id_a, 0) + 1
                    counts[id_b] = counts.get(id_b, 0) + 1
            pairs_needed: List[Tuple[str, str]] = []
            for nid in new_ids:
                needed = self.cfg.matches_per_round - counts.get(nid, 0)
                if needed <= 0:
                    continue
                opponents = [i for i in item_ids if i != nid]
                self.rng.shuffle(opponents)
                for opp in opponents[:needed]:
                    pairs_needed.append((nid, opp))
            if not pairs_needed:
                continue
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            id_to_circle_first: Dict[str, bool] = {}
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, (id_a, id_b) in enumerate(pairs_needed):
                    raw_ident = f"catchup|{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    sha8 = hashlib.sha1(raw_ident.encode()).hexdigest()[:8]
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    id_to_circle_first[sha8] = circle_first_flag
                    prompts.append(
                        self.template.render(
                            entry_circle=texts_by_id[id_a],
                            entry_square=texts_by_id[id_b],
                            attributes=attr_def_map,
                            additional_instructions=self.cfg.additional_instructions or "",
                            modality=self.cfg.modality,
                            circle_first=circle_first_flag,
                        )
                    )
                    ids.append(sha8)
                    meta_map[sha8] = (batch_idx, pair_idx, id_a, id_b)
                    if images_by_id:
                        imgs = []
                        ia = images_by_id.get(id_a, [])
                        ib = images_by_id.get(id_b, [])
                        if circle_first_flag:
                            if ia:
                                imgs.extend(ia)
                            if ib:
                                imgs.extend(ib)
                        else:
                            if ib:
                                imgs.extend(ib)
                            if ia:
                                imgs.extend(ia)
                        if imgs:
                            pair_images[sha8] = imgs
                    if audio_by_id:
                        auds = []
                        aa = audio_by_id.get(id_a, [])
                        ab = audio_by_id.get(id_b, [])
                        if circle_first_flag:
                            if aa:
                                auds.extend(aa)
                            if ab:
                                auds.extend(ab)
                        else:
                            if ab:
                                auds.extend(ab)
                            if aa:
                                auds.extend(aa)
                        if auds:
                            pair_audio[sha8] = auds
            if not prompts:
                continue
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                save_path=round_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            resp_df["Batch"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[0]
            )
            resp_df["Pair"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[1]
            )
            resp_df["IdA"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[2]
            )
            resp_df["IdB"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[3]
            )
            resp_df.to_csv(round_path, index=False)

            async def _coerce_dict(raw: Any) -> Dict[str, Any]:
                obj = await safest_json(raw)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    obj2 = await safest_json(obj)
                    if isinstance(obj2, dict):
                        return obj2
                if isinstance(obj, list) and obj:
                    inner = await safest_json(obj[0])
                    if isinstance(inner, dict):
                        return inner
                return {}

            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                meta = meta_map.get(str(ident))
                if not meta:
                    continue
                batch_idx, _, id_a, id_b = meta
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    if v.startswith(("cir", "c", "left", "text a")):
                        history_pairs[real_attr].append((id_a, id_b))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        history_pairs[real_attr].append((id_b, id_a))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        continue
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                if len(outcomes) == 0:
                    continue
                bt_scores, n_ij, p_ij = self._fit_bt(
                    item_ids=item_ids,
                    outcomes=outcomes,
                    pseudo=self.cfg.learning_rate,
                    max_iter=self._MAX_ITER,
                    tol=self._TOL,
                    return_info=True,
                )
                for i in item_ids:
                    ratings[i][attr] = bt_scores[i]
                if self.cfg.compute_se:
                    s_vec = np.array([bt_scores[i] for i in item_ids])
                    se_vec = self._bt_standard_errors(
                        s=s_vec,
                        n_ij=n_ij,
                        p_ij=p_ij,
                        ridge=self._SE_RIDGE,
                    )
                    for i, se_val in zip(item_ids, se_vec):
                        se_store[attr][i] = float(se_val)
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
            if self.cfg.compute_se:
                for i in item_ids:
                    if se_agg_counts[i] > 0:
                        se_agg_next[i] /= se_agg_counts[i]
                    else:
                        se_agg_next[i] = 1.0
                self._last_se_agg = se_agg_next
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
            _write_checkpoint()

    async def _run_recursive(
        self,
        df: pd.DataFrame,
        text_column: str,
        *,
        id_column: Optional[str],
        reset_files: bool,
        **kwargs: Any,
    ) -> pd.DataFrame:
        attr_dict = self._attributes_as_dict()
        attr_list = list(attr_dict.keys())
        if not attr_list:
            raise ValueError("No attributes provided for ranking")
        cut_attr = self.cfg.recursive_cut_attr or attr_list[0]
        if cut_attr not in attr_list:
            raise ValueError(
                f"recursive_cut_attr '{self.cfg.recursive_cut_attr}' not present in attributes"
            )
        cut_side = (self.cfg.recursive_cut_side or "top").lower()
        if cut_side not in {"top", "bottom"}:
            raise ValueError("recursive_cut_side must be 'top' or 'bottom'")

        work_df = df.reset_index(drop=True).copy()
        if id_column is not None:
            if id_column not in work_df.columns:
                raise ValueError(f"id_column '{id_column}' not found in DataFrame")
            work_df["identifier"] = work_df[id_column].astype(str)
        else:
            work_df["identifier"] = work_df[text_column].astype(str).map(
                lambda x: hashlib.sha1(x.encode()).hexdigest()[:8]
            )
        if text_column != "text":
            work_df = work_df.rename(columns={text_column: "text"})
        rewrite_col = self.cfg.recursive_rewrite_text_col or "text"
        if rewrite_col not in work_df.columns:
            work_df[rewrite_col] = work_df["text"]
        work_df["identifier"] = work_df["identifier"].astype(str)

        base_folder = os.path.join(
            self.cfg.save_dir, f"{self.cfg.file_name}_recursive"
        )
        os.makedirs(base_folder, exist_ok=True)

        cumulative_scores: Dict[str, Dict[str, float]] = {
            attr: {ident: 0.0 for ident in work_df["identifier"]}
            for attr in attr_list
        }
        exit_stage: Dict[str, Optional[int]] = {
            ident: None for ident in work_df["identifier"]
        }
        stage_dfs: List[pd.DataFrame] = []
        current_ids = list(work_df["identifier"])
        all_ids = list(current_ids)

        def _select_next_ids(active_ids: Sequence[str]) -> List[str]:
            n = len(active_ids)
            if n <= self.cfg.recursive_min_remaining:
                return list(active_ids)
            keep_n = max(
                int(math.ceil(n * self.cfg.recursive_fraction)),
                self.cfg.recursive_min_remaining,
            )
            series = pd.Series(
                {i: cumulative_scores[cut_attr][i] for i in active_ids},
                name="cumulative",
            )
            ascending = cut_side == "bottom"
            ranked = series.sort_values(ascending=ascending)
            return ranked.head(keep_n).index.tolist()

        def _update_cumulative(stage_df: pd.DataFrame) -> None:
            for attr in attr_list:
                raw_col = f"{attr}_raw"
                col_name = raw_col if raw_col in stage_df.columns else attr
                if col_name not in stage_df.columns:
                    continue
                for ident, value in zip(stage_df["identifier"], stage_df[col_name]):
                    try:
                        cumulative_scores[attr][str(ident)] += float(value)
                    except Exception:
                        continue

        def _maybe_rewrite_texts(
            df_local: pd.DataFrame,
            ids_to_keep: Sequence[str],
            stage_idx: int,
        ) -> pd.DataFrame:
            if self.cfg.recursive_rewrite_func is None:
                return df_local
            mask = df_local["identifier"].isin(ids_to_keep)
            rewritten: List[str] = []
            for _, row in df_local[mask].iterrows():
                rewritten.append(
                    self.cfg.recursive_rewrite_func(
                        row[self.cfg.recursive_rewrite_text_col],
                        row["identifier"],
                        stage_idx,
                    )
                )
            df_local.loc[mask, self.cfg.recursive_rewrite_text_col] = rewritten
            if (
                self.cfg.recursive_rewrite_text_col != "text"
                and "text" in df_local.columns
            ):
                df_local.loc[mask, "text"] = df_local.loc[
                    mask, self.cfg.recursive_rewrite_text_col
                ]
            return df_local

        stage_idx = 0
        final_stage_df: Optional[pd.DataFrame] = None

        while current_ids:
            stage_idx += 1
            n_current = len(current_ids)
            is_final_stage = False
            if n_current <= self.cfg.recursive_min_remaining:
                is_final_stage = True
            else:
                next_keep = max(
                    int(math.ceil(n_current * self.cfg.recursive_fraction)),
                    self.cfg.recursive_min_remaining,
                )
                if next_keep <= self.cfg.recursive_min_remaining:
                    is_final_stage = True

            stage_rounds = self.cfg.n_rounds
            if is_final_stage:
                stage_rounds = max(
                    1, stage_rounds * self.cfg.recursive_final_round_multiplier
                )

            stage_folder = os.path.join(base_folder, f"stage{stage_idx}")
            os.makedirs(stage_folder, exist_ok=True)
            stage_cfg = copy.deepcopy(self.cfg)
            stage_cfg.recursive = False
            stage_cfg.recursive_rate_first_round = False
            stage_cfg.save_dir = stage_folder
            stage_cfg.n_rounds = stage_rounds
            stage_cfg.file_name = self.cfg.file_name
            stage_cfg.rate_kwargs = dict(self.cfg.rate_kwargs)

            stage_df_in = work_df[work_df["identifier"].isin(current_ids)].copy()

            if stage_idx == 1 and self.cfg.recursive_rate_first_round:
                stage_df_out = await self._run_rate_pass(
                    stage_df_in,
                    column_name="text",
                    save_dir=stage_folder,
                    file_name=f"stage{stage_idx}_ratings.csv",
                    reset_files=reset_files,
                    runtime_kwargs=kwargs,
                )
            else:
                stage_ranker = Rank(stage_cfg, template=self.template)
                stage_df_out = await stage_ranker.run(
                    stage_df_in,
                    column_name="text",
                    id_column="identifier",
                    reset_files=reset_files,
                    **kwargs,
                )

            if self.cfg.recursive_keep_stage_columns:
                keep_cols = [c for c in stage_df_out.columns if c != "text"]
                if self.cfg.recursive_add_stage_suffix:
                    renamed = {
                        c: f"stage{stage_idx}_{c}"
                        for c in keep_cols
                        if c != "identifier"
                    }
                    stage_df_clean = stage_df_out.rename(columns=renamed)
                else:
                    stage_df_clean = stage_df_out.copy()
                stage_dfs.append(stage_df_clean.drop(columns=["text"], errors="ignore"))

            _update_cumulative(stage_df_out)

            if is_final_stage:
                for ident in current_ids:
                    exit_stage[ident] = stage_idx
                final_stage_df = stage_df_out
                break

            next_ids = _select_next_ids(current_ids)
            removed = set(current_ids) - set(next_ids)
            for ident in removed:
                exit_stage[ident] = stage_idx
            work_df = _maybe_rewrite_texts(work_df, next_ids, stage_idx)
            current_ids = next_ids

        if final_stage_df is None:
            final_stage_df = work_df[work_df["identifier"].isin(current_ids)].copy()

        cum_rows = []
        for ident in all_ids:
            row = {"identifier": ident}
            for attr in attr_list:
                row[f"cumulative_{attr}"] = cumulative_scores[attr][ident]
            cum_rows.append(row)
        cum_df = pd.DataFrame(cum_rows)

        exit_df = pd.DataFrame(
            {"identifier": list(exit_stage.keys()), "exit_stage": list(exit_stage.values())}
        )

        final_cols = [c for c in final_stage_df.columns if c != "identifier"]
        final_raw = final_stage_df.rename(
            columns={
                c: ("final_text" if c == "text" else f"final_{c}")
                for c in final_cols
            }
        )

        latest_text_df = work_df[["identifier", "text"]].copy()

        out = (
            cum_df.merge(exit_df, on="identifier", how="left")
            .merge(latest_text_df, on="identifier", how="left")
            .merge(final_raw, on="identifier", how="left")
        )
        if self.cfg.recursive_keep_stage_columns and stage_dfs:
            for sdf in stage_dfs:
                out = out.merge(sdf, on="identifier", how="left")

        prefixed_cum = [c for c in out.columns if c.startswith("cumulative_")]
        prefixed_final = [c for c in out.columns if c.startswith("final_")]
        ordered_cols = ["identifier", "text", "exit_stage"] + prefixed_cum + prefixed_final
        remaining = [c for c in out.columns if c not in ordered_cols]
        out = out[ordered_cols + remaining]

        final_path = os.path.join(base_folder, "recursive_final.csv")
        out.to_csv(final_path, index=False)
        return out

    # ------------------------------------------------------------------
    # Main ranking loop
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        id_column: Optional[str] = None,
        reset_files: bool = False,
        n_runs: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Execute the ranking procedure.

        Parameters
        ----------
        df:
            Input DataFrame containing the passages to be ranked.
        column_name:
            Name of the column in ``df`` that holds the text for each
            passage.
        id_column:
            Optional name of a column that contains stable identifiers
            for each row. When provided, these identifiers are used to
            track passages across rounds instead of hashing the text
            itself.  Supplying ``id_column`` is recommended when texts
            may be rewritten between stages (e.g., during recursive
            runs).
        reset_files:
            If ``True``, ignore any previously saved results and
            recompute the rankings.  Otherwise, if the final output
            file already exists on disk it will be loaded and returned
            immediately.
        n_runs:
            Deprecated/ignored parameter provided for compatibility
            with :class:`Rate`. When supplied, a message is printed
            noting that ``n_rounds`` controls the number of iterations
            and that ``n_runs`` has no effect.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`get_all_responses`.  When ``initial_rating_pass`` is
            enabled these arguments are also forwarded to the rating
            stage.  Useful for passing through authentication tokens or
            tracing settings.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with one row per input passage.  For each
            attribute the DataFrame contains a ``"<attribute>"`` column
            holding the z‑score (or the raw Bradley–Terry score when
            ``add_zscore`` is ``False``), a ``"<attribute>_raw"`` column
            with the centred Bradley–Terry estimate, and—when
            ``compute_se`` is ``True``—an optional
            ``"<attribute>_se"`` column with the standard error.  The
            DataFrame is also written to ``save_dir``.
        """
        kwargs.setdefault("web_search", self.cfg.modality == "web")
        if self.cfg.recursive:
            return await self._run_recursive(
                df,
                column_name,
                id_column=id_column,
                reset_files=reset_files,
                **kwargs,
            )

        # prepare file paths
        base_name = os.path.splitext(self.cfg.file_name)[0]
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")
        attr_path = os.path.join(self.cfg.save_dir, f"{base_name}_attrs.json")
        if n_runs is not None:
            print(
                "Parameter 'n_runs' is ignored. Use 'n_rounds' to control the number of iterations. "
                f"Current n_rounds={self.cfg.n_rounds}."
            )

        df_proc = df.reset_index(drop=True).copy()
        if id_column is not None:
            if id_column not in df_proc.columns:
                raise ValueError(f"id_column '{id_column}' not found in DataFrame")
            df_proc["_id"] = df_proc[id_column].astype(str)
        else:
            # assign a stable identifier per row using an sha1 hash
            df_proc["_id"] = (
                df_proc[column_name]
                .astype(str)
                .map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:8])
            )
        if reset_files and os.path.exists(attr_path):
            try:
                os.remove(attr_path)
            except Exception:
                pass
        if os.path.exists(attr_path):
            try:
                with open(attr_path) as f:
                    saved_attrs = json.load(f)
                if saved_attrs != self.cfg.attributes:
                    print(
                        "[Rank] Loading existing attributes from save directory. If you want to use different attributes, set reset_files=True or use a different save_dir."
                    )
                    print(saved_attrs)
                    self.cfg.attributes = saved_attrs
            except Exception:
                pass
        else:
            try:
                with open(attr_path, "w") as f:
                    json.dump(self.cfg.attributes, f, indent=2)
            except Exception:
                pass
        # Determine how many rounds have already been processed when
        # `reset_files` is False.  We look for files named
        # ``<base_name>_round<k>.csv`` to infer progress.  If a final
        # checkpoint exists for the last round, reuse it; otherwise we
        # resume from the next incomplete round.  When ``reset_files``
        # is ``True``, all progress is ignored and the computation
        # restarts from round 0.
        start_round = 0
        existing_rounds: List[int] = []
        if not reset_files:
            try:
                for fname in os.listdir(self.cfg.save_dir):
                    if fname.startswith(f"{base_name}_round") and fname.endswith(
                        ".csv"
                    ):
                        try:
                            idx_str = fname[
                                len(base_name) + 6 : -4
                            ]  # len("_round") == 6
                            rnd_idx = int(idx_str)
                            existing_rounds.append(rnd_idx)
                        except Exception:
                            continue
            except Exception:
                existing_rounds = []
        if existing_rounds:
            last_completed = max(existing_rounds)
            if os.path.exists(final_path):
                try:
                    final_df = pd.read_csv(final_path)
                    identifier_col = (
                        id_column
                        if id_column and id_column in final_df.columns
                        else column_name
                    )
                    if identifier_col not in final_df.columns:
                        raise ValueError(
                            "Existing ranking output is missing identifier column "
                            f"'{identifier_col}'."
                        )
                    if id_column:
                        final_ids = set(final_df[identifier_col].astype(str))
                    else:
                        final_ids = set(
                            final_df[identifier_col]
                            .astype(str)
                            .map(lambda x: hashlib.sha1(x.encode()).hexdigest()[:8])
                        )
                    if last_completed >= self.cfg.n_rounds - 1 and set(df_proc["_id"]) <= final_ids:
                        return final_df
                except Exception:
                    pass
            start_round = last_completed + 1
        # extract contents and build lookup
        if self.cfg.modality in {"image", "audio"}:
            texts = list(zip(df_proc["_id"], ["" for _ in df_proc[column_name]]))
        else:
            texts = list(zip(df_proc["_id"], df_proc[column_name].astype(str)))
        texts_by_id = {i: t for i, t in texts}
        item_ids = [i for i, _ in texts]

        images_by_id: Dict[str, List[str]] = {}
        audio_by_id: Dict[str, List[Dict[str, str]]] = {}
        if self.cfg.modality == "image":
            for rid, imgs in zip(df_proc["_id"], df_proc[column_name]):
                encoded = load_image_inputs(imgs)
                if encoded:
                    images_by_id[rid] = encoded
        elif self.cfg.modality == "audio":
            for rid, auds in zip(df_proc["_id"], df_proc[column_name]):
                encoded = load_audio_inputs(auds)
                if encoded:
                    audio_by_id[rid] = encoded
        # derive list of attributes
        if isinstance(self.cfg.attributes, dict):
            attr_keys = list(self.cfg.attributes.keys())
        else:
            attr_keys = list(self.cfg.attributes)
        # initialise ratings for each item/attribute
        ratings: Dict[str, Dict[str, float]] = {
            i: {a: 0.0 for a in attr_keys} for i in item_ids
        }
        rate_seed: Dict[str, Dict[str, float]] = {}
        if self.cfg.initial_rating_pass and attr_keys:
            rate_dir = os.path.join(self.cfg.save_dir, f"{base_name}_initial_rate")
            os.makedirs(rate_dir, exist_ok=True)
            rate_df = await self._run_rate_pass(
                df_proc,
                column_name,
                save_dir=rate_dir,
                file_name=f"{base_name}_initial_rate.csv",
                reset_files=reset_files,
                runtime_kwargs=kwargs,
            )
            rate_seed = self._seed_ratings_from_rate(
                rate_df,
                id_column=id_column,
                text_column=column_name,
                item_ids=item_ids,
                attr_keys=attr_keys,
            )
            for item_id, attr_map in rate_seed.items():
                for attr, val in attr_map.items():
                    ratings[item_id][attr] = val
        has_seed_ratings = bool(rate_seed)
        # maintain a history of pairwise outcomes for each attribute
        history_pairs: Dict[str, List[Tuple[str, str]]] = {a: [] for a in attr_keys}
        # store per‑attribute standard errors across items
        se_store: Dict[str, Dict[str, float]] = {
            a: {i: np.nan for i in item_ids} for a in attr_keys
        }
        # Define attribute batches once to reuse across replay and new rounds
        attr_batches: List[List[str]] = [
            attr_keys[i : i + self.cfg.n_attributes_per_run]
            for i in range(0, len(attr_keys), self.cfg.n_attributes_per_run)
        ]


        # Helper function to write the current results to the final CSV.  This
        # builds the output DataFrame from the current ``df_proc`` and
        # ``ratings``/``se_store``/``zscores`` and writes it to
        # ``final_path``.
        def _write_checkpoint() -> None:
            # Compute z‑scores for each attribute if required
            zscores_local: Dict[str, Dict[str, float]] = {}
            if self.cfg.add_zscore:
                for attr in attr_keys:
                    vals = np.array([ratings[i][attr] for i in item_ids])
                    mean = vals.mean()
                    std = vals.std(ddof=0)
                    if std == 0:
                        zscores_local[attr] = {i: 0.0 for i in item_ids}
                    else:
                        zscores_local[attr] = {
                            i: float((ratings[i][attr] - mean) / std) for i in item_ids
                        }
            # Merge computed results back into the original DataFrame copy.
            for attr in attr_keys:
                raw_col = f"{attr}_raw"
                # ratings
                val_map = {i: ratings[i][attr] for i in item_ids}
                df_proc[raw_col] = df_proc["_id"].map(val_map)
                # standard errors
                if self.cfg.compute_se:
                    se_map = {i: se_store[attr].get(i, np.nan) for i in item_ids}
                    df_proc[f"{attr}_se"] = df_proc["_id"].map(se_map)
                # z‑scores (or fall back to raw scores when disabled)
                if self.cfg.add_zscore:
                    z_map = zscores_local.get(attr, {i: np.nan for i in item_ids})
                    df_proc[attr] = df_proc["_id"].map(z_map)
                else:
                    df_proc[attr] = df_proc[raw_col]
            # Reorder columns: original user columns first (excluding the internal ``_id``),
            # then for each attribute the z‑score column, followed by raw scores and standard errors.
            original_cols = [
                c for c in df.columns
            ]  # preserve the order provided by the user
            new_cols: List[str] = []
            for attr in attr_keys:
                new_cols.append(attr)
                new_cols.append(f"{attr}_raw")
                if self.cfg.compute_se:
                    new_cols.append(f"{attr}_se")
            final_cols = original_cols + new_cols
            final_cols = [c for c in final_cols if c in df_proc.columns]
            df_out_local = df_proc[final_cols].copy()
            # Write the final results to disk in CSV format.  Using CSV avoids
            # Excel row limits and unnecessary overhead.
            df_out_local.to_csv(final_path, index=False)

        # If there are completed rounds and we're resuming, replay them to
        # reconstruct the ratings and uncertainties.  After each replayed
        # round we write a checkpoint to ``final_path``.
        if start_round > 0:
            for replay_rnd in range(start_round):
                round_path = os.path.join(
                    self.cfg.save_dir, f"{base_name}_round{replay_rnd}.csv"
                )
                if not os.path.exists(round_path):
                    break
                try:
                    # Load existing responses for this round
                    df_round = pd.read_csv(round_path)
                    df_round["Response"] = df_round["Response"].apply(
                        lambda x: None if pd.isna(x) else x
                    )
                except Exception:
                    continue

                # Parse each response to build history_pairs
                async def _coerce_dict_replay(raw: Any) -> Dict[str, Any]:
                    obj = await safest_json(raw)
                    if isinstance(obj, dict):
                        return obj
                    if isinstance(obj, str):
                        obj2 = await safest_json(obj)
                        if isinstance(obj2, dict):
                            return obj2
                    if isinstance(obj, list) and obj:
                        inner = await safest_json(obj[0])
                        if isinstance(inner, dict):
                            return inner
                    return {}

                if {"Batch", "IdA", "IdB"}.issubset(df_round.columns):
                    for batch_idx_raw, id_a, id_b, resp_raw in zip(
                        df_round["Batch"],
                        df_round["IdA"],
                        df_round["IdB"],
                        df_round["Response"],
                    ):
                        batch_idx = int(batch_idx_raw)
                        batch = attr_batches[batch_idx]
                        batch_attr_map = {str(k).strip().lower(): k for k in batch}
                        safe_obj = await _coerce_dict_replay(resp_raw)
                        if not safe_obj:
                            continue
                        for attr_raw, winner_raw in safe_obj.items():
                            attr_key_l = str(attr_raw).strip().lower()
                            if attr_key_l not in batch_attr_map:
                                continue
                            real_attr = batch_attr_map[attr_key_l]
                            val = winner_raw
                            if isinstance(val, dict) and "winner" in val:
                                val = val.get("winner")
                            if isinstance(val, str):
                                v = val.strip().lower()
                            else:
                                v = ""
                            if v.startswith(("cir", "c", "left", "text a")):
                                history_pairs[real_attr].append((id_a, id_b))
                            elif v.startswith(("squ", "b", "right", "text b")):
                                history_pairs[real_attr].append((id_b, id_a))
                            elif v.startswith("draw") or v.startswith("insufficient"):
                                history_pairs[real_attr].append((id_a, id_b))
                                history_pairs[real_attr].append((id_b, id_a))
                            else:
                                continue
                else:
                    for ident, resp_raw in zip(
                        df_round["Identifier"], df_round["Response"]
                    ):
                        parts = str(ident).split("|")
                        if len(parts) != 5:
                            continue
                        _, batch_idx_str, _, id_a, id_b = parts
                        batch_idx = int(batch_idx_str)
                        batch = attr_batches[batch_idx]
                        batch_attr_map = {str(k).strip().lower(): k for k in batch}
                        safe_obj = await _coerce_dict_replay(resp_raw)
                        if not safe_obj:
                            continue
                        for attr_raw, winner_raw in safe_obj.items():
                            attr_key_l = str(attr_raw).strip().lower()
                            if attr_key_l not in batch_attr_map:
                                continue
                            real_attr = batch_attr_map[attr_key_l]
                            val = winner_raw
                            if isinstance(val, dict) and "winner" in val:
                                val = val.get("winner")
                            if isinstance(val, str):
                                v = val.strip().lower()
                            else:
                                v = ""
                            if v.startswith(("cir", "c", "left", "text a")):
                                history_pairs[real_attr].append((id_a, id_b))
                            elif v.startswith(("squ", "b", "right", "text b")):
                                history_pairs[real_attr].append((id_b, id_a))
                            elif v.startswith("draw") or v.startswith("insufficient"):
                                history_pairs[real_attr].append((id_a, id_b))
                                history_pairs[real_attr].append((id_b, id_a))
                            else:
                                continue
                # After parsing all pairs for this round, update ratings
                se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
                se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
                for attr in attr_keys:
                    outcomes = history_pairs[attr]
                    if len(outcomes) == 0:
                        continue
                    bt_scores, n_ij, p_ij = self._fit_bt(
                        item_ids=item_ids,
                        outcomes=outcomes,
                        pseudo=self.cfg.learning_rate,
                        max_iter=self._MAX_ITER,
                        tol=self._TOL,
                        return_info=True,
                    )
                    for i in item_ids:
                        ratings[i][attr] = bt_scores[i]
                    if self.cfg.compute_se:
                        s_vec = np.array([bt_scores[i] for i in item_ids])
                        se_vec = self._bt_standard_errors(
                            s=s_vec,
                            n_ij=n_ij,
                            p_ij=p_ij,
                            ridge=self._SE_RIDGE,
                        )
                        for i, se_val in zip(item_ids, se_vec):
                            se_store[attr][i] = float(se_val)
                            se_agg_next[i] += float(se_val)
                            se_agg_counts[i] += 1
                if self.cfg.compute_se:
                    for i in item_ids:
                        if se_agg_counts[i] > 0:
                            se_agg_next[i] /= se_agg_counts[i]
                        else:
                            se_agg_next[i] = 1.0
                    self._last_se_agg = se_agg_next
                # Centre ratings to zero mean for each attribute
                for attr in attr_keys:
                    vals = [ratings[i][attr] for i in item_ids]
                    mean_val = float(np.mean(vals))
                    for i in item_ids:
                        ratings[i][attr] -= mean_val
                # Write checkpoint after this replayed round
                _write_checkpoint()

        # Determine if any new items were added and need to catch up on existing rounds
        seen_ids: set[str] = set()
        for pair_list in history_pairs.values():
            for a, b in pair_list:
                seen_ids.add(a)
                seen_ids.add(b)
        new_ids = [i for i in item_ids if i not in seen_ids]
        await self._catch_up_existing_rounds(
            new_ids=new_ids,
            round_indices=list(range(start_round)),
            item_ids=item_ids,
            texts_by_id=texts_by_id,
            images_by_id=images_by_id,
            audio_by_id=audio_by_id,
            attr_batches=attr_batches,
            attr_keys=attr_keys,
            history_pairs=history_pairs,
            ratings=ratings,
            se_store=se_store,
            base_name=base_name,
            df_proc=df_proc,
            _write_checkpoint=_write_checkpoint,
            current_ratings=None,
            se_agg_local=self._last_se_agg,
            reset_files=reset_files,
            **kwargs,
        )

        # Now proceed with new rounds starting from ``start_round``
        for rnd in range(start_round, self.cfg.n_rounds):
            # aggregate current ratings across attributes for pairing
            current_agg = {
                i: float(np.mean(list(ratings[i].values()))) for i in item_ids
            }
            se_agg_local = self._last_se_agg
            use_current = rnd > 0 or start_round > 0 or has_seed_ratings
            se_source = se_agg_local if (rnd > 0 or start_round > 0 or se_agg_local is not None) else None
            pairs = self._generate_pairs(
                item_ids=item_ids,
                texts_by_id=texts_by_id,
                current_ratings=current_agg if use_current else None,
                se_agg=se_source,
            )
            if not pairs:
                break
            prompts: List[str] = []
            ids: List[str] = []
            pair_images: Dict[str, List[str]] = {}
            pair_audio: Dict[str, List[Dict[str, str]]] = {}
            meta_map: Dict[str, Tuple[int, int, str, str]] = {}
            id_to_circle_first: Dict[str, bool] = {}
            for batch_idx, batch in enumerate(attr_batches):
                attr_def_map = (
                    {a: self.cfg.attributes[a] for a in batch}
                    if isinstance(self.cfg.attributes, dict)
                    else {a: "" for a in batch}
                )
                for pair_idx, ((id_a, t_a), (id_b, t_b)) in enumerate(pairs):
                    raw_ident = f"{rnd}|{batch_idx}|{pair_idx}|{id_a}|{id_b}"
                    sha8 = hashlib.sha1(raw_ident.encode()).hexdigest()[:8]
                    circle_first_flag = (
                        self.cfg.circle_first
                        if self.cfg.circle_first is not None
                        else self.rng.random() < 0.5
                    )
                    id_to_circle_first[sha8] = circle_first_flag
                    prompts.append(
                        self.template.render(
                            entry_circle=t_a,
                            entry_square=t_b,
                            attributes=attr_def_map,
                            additional_instructions=self.cfg.additional_instructions
                            or "",
                            modality=self.cfg.modality,
                            circle_first=circle_first_flag,
                        )
                    )
                    ids.append(sha8)
                    meta_map[sha8] = (batch_idx, pair_idx, id_a, id_b)
                    if images_by_id:
                        imgs = []
                        ia = images_by_id.get(id_a, [])
                        ib = images_by_id.get(id_b, [])
                        if circle_first_flag:
                            if ia:
                                imgs.extend(ia)
                            if ib:
                                imgs.extend(ib)
                        else:
                            if ib:
                                imgs.extend(ib)
                            if ia:
                                imgs.extend(ia)
                        if imgs:
                            pair_images[sha8] = imgs
                    if audio_by_id:
                        auds = []
                        aa = audio_by_id.get(id_a, [])
                        ab = audio_by_id.get(id_b, [])
                        if circle_first_flag:
                            if aa:
                                auds.extend(aa)
                            if ab:
                                auds.extend(ab)
                        else:
                            if ab:
                                auds.extend(ab)
                            if aa:
                                auds.extend(aa)
                        if auds:
                            pair_audio[sha8] = auds
            # obtain responses from the language model for this round
            round_path = os.path.join(self.cfg.save_dir, f"{base_name}_round{rnd}.csv")
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=pair_images or None,
                prompt_audio=pair_audio or None,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                json_mode=self.cfg.modality != "audio",
                save_path=round_path,
                reset_files=reset_files,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                max_retries=1,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            # attach metadata columns and overwrite the round CSV
            resp_df["Batch"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[0]
            )
            resp_df["Pair"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[1]
            )
            resp_df["IdA"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[2]
            )
            resp_df["IdB"] = resp_df.Identifier.map(
                lambda x: meta_map.get(str(x), (np.nan, np.nan, "", ""))[3]
            )
            resp_df.to_csv(round_path, index=False)

            # parse each response
            # reuse the _coerce_dict function defined in the original implementation
            async def _coerce_dict(raw: Any) -> Dict[str, Any]:
                obj = await safest_json(raw)
                if isinstance(obj, dict):
                    return obj
                if isinstance(obj, str):
                    obj2 = await safest_json(obj)
                    if isinstance(obj2, dict):
                        return obj2
                if isinstance(obj, list) and obj:
                    inner = await safest_json(obj[0])
                    if isinstance(inner, dict):
                        return inner
                return {}

            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                meta = meta_map.get(str(ident))
                if not meta:
                    continue
                batch_idx, _, id_a, id_b = meta
                safe_obj = await _coerce_dict(resp)
                if not safe_obj:
                    continue
                batch = attr_batches[batch_idx]
                batch_attr_map = {str(k).strip().lower(): k for k in batch}
                for attr_raw, winner_raw in safe_obj.items():
                    attr_key_l = str(attr_raw).strip().lower()
                    if attr_key_l not in batch_attr_map:
                        continue
                    real_attr = batch_attr_map[attr_key_l]
                    val = winner_raw
                    if isinstance(val, dict) and "winner" in val:
                        val = val.get("winner")
                    if isinstance(val, str):
                        v = val.strip().lower()
                    else:
                        v = ""
                    if v.startswith(("cir", "c", "left", "text a")):
                        history_pairs[real_attr].append((id_a, id_b))
                    elif v.startswith(("squ", "b", "right", "text b")):
                        history_pairs[real_attr].append((id_b, id_a))
                    elif v.startswith("draw") or v.startswith("insufficient"):
                        history_pairs[real_attr].append((id_a, id_b))
                        history_pairs[real_attr].append((id_b, id_a))
                    else:
                        continue
            # update ratings using the BT model for this round
            se_agg_next: Dict[str, float] = {i: 0.0 for i in item_ids}
            se_agg_counts: Dict[str, int] = {i: 0 for i in item_ids}
            for attr in attr_keys:
                outcomes = history_pairs[attr]
                if len(outcomes) == 0:
                    continue
                bt_scores, n_ij, p_ij = self._fit_bt(
                    item_ids=item_ids,
                    outcomes=outcomes,
                    pseudo=self.cfg.learning_rate,
                    max_iter=self._MAX_ITER,
                    tol=self._TOL,
                    return_info=True,
                )
                for i in item_ids:
                    ratings[i][attr] = bt_scores[i]
                if self.cfg.compute_se:
                    s_vec = np.array([bt_scores[i] for i in item_ids])
                    se_vec = self._bt_standard_errors(
                        s=s_vec,
                        n_ij=n_ij,
                        p_ij=p_ij,
                        ridge=self._SE_RIDGE,
                    )
                    for i, se_val in zip(item_ids, se_vec):
                        se_store[attr][i] = float(se_val)
                        se_agg_next[i] += float(se_val)
                        se_agg_counts[i] += 1
            if self.cfg.compute_se:
                for i in item_ids:
                    if se_agg_counts[i] > 0:
                        se_agg_next[i] /= se_agg_counts[i]
                    else:
                        se_agg_next[i] = 1.0
                self._last_se_agg = se_agg_next
            # Centre ratings to zero mean for each attribute
            for attr in attr_keys:
                vals = [ratings[i][attr] for i in item_ids]
                mean_val = float(np.mean(vals))
                for i in item_ids:
                    ratings[i][attr] -= mean_val
            # Write checkpoint after this new round
            _write_checkpoint()
        # After processing all rounds, return the final DataFrame
        # The checkpoint has already been written in the final iteration
        return pd.read_csv(final_path)

