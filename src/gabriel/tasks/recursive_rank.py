from __future__ import annotations

import os
import math
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Any

import pandas as pd

from .rank import Rank, RankConfig
from .rate import Rate, RateConfig


@dataclass
class RecursiveRankConfig:
    """Configuration for :class:`RecursiveRank`.

    Parameters
    ----------
    base_cfg:
        Baseline :class:`RankConfig` used for each recursion stage.
        A deep copy of this configuration is created for every stage
        with stage-specific adjustments (e.g., number of rounds,
        output directory).
    cut_attr:
        Attribute to use when selecting the fraction of items to keep
        after each stage.  If ``None`` the first attribute from
        ``base_cfg.attributes`` is used.
    cut_side:
        ``"top"`` retains the highest scoring items while
        ``"bottom"`` retains the lowest.
    fraction:
        Fraction of items to keep after each stage.
    min_remaining:
        Minimum number of items allowed in the final stage.  If a cut
        would reduce the pool below this number, one last stage is run
        with ``final_round_multiplier`` applied to ``n_rounds`` and the
        recursion terminates.
    final_round_multiplier:
        How much to multiply ``n_rounds`` for the final stage.
    rewrite_func:
        Optional function applied to the surviving texts between stages.
        Signature: ``(text, identifier, stage_idx) -> new_text``.
    rewrite_text_col:
        Column name in which rewritten text should be stored.
    rate_first_round:
        If ``True`` the first stage is a simple :class:`Rate` pass on
        all items.  The resulting scores seed the cumulative ratings
        used for subsequent ranking stages.
    """

    base_cfg: RankConfig
    cut_attr: Optional[str] = None
    cut_side: str = "top"
    fraction: float = 1.0 / 3.0
    min_remaining: int = 30
    final_round_multiplier: int = 3
    rewrite_func: Optional[Callable[[str, str, int], str]] = None
    rewrite_text_col: str = "text"
    rate_first_round: bool = False

    # Internal / housekeeping
    keep_stage_columns: bool = True
    add_stage_suffix: bool = True


class RecursiveRank:
    """Recursively rank items using repeated calls to :class:`Rank`."""

    def __init__(self, cfg: RecursiveRankConfig) -> None:
        self.cfg = cfg

        if self.cfg.cut_side not in ("top", "bottom"):
            raise ValueError("cut_side must be 'top' or 'bottom'")

        if isinstance(self.cfg.base_cfg.attributes, dict):
            self._attr_list = list(self.cfg.base_cfg.attributes.keys())
        else:
            self._attr_list = list(self.cfg.base_cfg.attributes)
        if not self._attr_list:
            raise ValueError("No attributes found in base_cfg.attributes")

        if self.cfg.cut_attr is None:
            self.cut_attr = self._attr_list[0]
        else:
            if self.cfg.cut_attr not in self._attr_list:
                raise ValueError(f"cut_attr '{self.cfg.cut_attr}' not in attributes")
            self.cut_attr = self.cfg.cut_attr

        self._cumulative_scores: Dict[str, Dict[str, float]] = {}
        self._stage_dfs: List[pd.DataFrame] = []
        self._base_folder = os.path.join(
            self.cfg.base_cfg.save_dir, f"{self.cfg.base_cfg.file_name}_recursive"
        )
        os.makedirs(self._base_folder, exist_ok=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _clone_cfg_for_stage(self, stage_idx: int, n_rounds: int) -> RankConfig:
        base = self.cfg.base_cfg
        new_cfg = copy.deepcopy(base)
        stage_folder = os.path.join(self._base_folder, f"stage{stage_idx}")
        os.makedirs(stage_folder, exist_ok=True)
        new_cfg.save_dir = stage_folder
        new_cfg.n_rounds = n_rounds
        return new_cfg

    def _init_cumulative_scores(self, ids: Sequence[str]) -> None:
        if self._cumulative_scores:
            return
        for attr in self._attr_list:
            self._cumulative_scores[attr] = {i: 0.0 for i in ids}

    def _update_cumulative(self, stage_df: pd.DataFrame) -> None:
        for attr in self._attr_list:
            if attr not in stage_df.columns:
                continue
            for i, val in zip(stage_df["identifier"], stage_df[attr]):
                self._cumulative_scores[attr][i] += float(val)

    def _get_rank_series(self, ids: Sequence[str]) -> pd.Series:
        data = {i: self._cumulative_scores[self.cut_attr][i] for i in ids}
        s = pd.Series(data, name="cumulative")
        return s.sort_values(ascending=(self.cfg.cut_side == "bottom"))

    def _select_next_ids(self, current_ids: Sequence[str]) -> List[str]:
        n = len(current_ids)
        if n <= self.cfg.min_remaining:
            return list(current_ids)
        keep_n = max(int(math.ceil(n * self.cfg.fraction)), self.cfg.min_remaining)
        ranked = self._get_rank_series(current_ids)
        return ranked.head(keep_n).index.tolist()

    def _maybe_rewrite_texts(
        self,
        df: pd.DataFrame,
        ids_to_keep: Sequence[str],
        stage_idx: int,
        text_col: str,
    ) -> pd.DataFrame:
        if self.cfg.rewrite_func is None:
            return df
        mask = df["identifier"].isin(ids_to_keep)
        rewritten = []
        for _, row in df[mask].iterrows():
            new_text = self.cfg.rewrite_func(row[text_col], row["identifier"], stage_idx)
            rewritten.append(new_text)
        df.loc[mask, self.cfg.rewrite_text_col] = rewritten
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def run(
        self, df: pd.DataFrame, text_col: str, id_col: str, *, reset_files: bool = False
    ) -> pd.DataFrame:
        work_df = df.copy()
        work_df[id_col] = work_df[id_col].astype(str)
        work_df = work_df.rename(columns={id_col: "identifier"})
        if text_col != "text":
            work_df = work_df.rename(columns={text_col: "text"})

        all_ids = work_df["identifier"].tolist()
        self._init_cumulative_scores(all_ids)

        exit_stage = {i: None for i in all_ids}
        current_ids = list(all_ids)
        stage_idx = 0

        while True:
            stage_idx += 1

            is_final_stage = False
            if len(current_ids) <= self.cfg.min_remaining:
                is_final_stage = True
            else:
                n = len(current_ids)
                next_keep_n = max(int(math.ceil(n * self.cfg.fraction)), self.cfg.min_remaining)
                if next_keep_n <= self.cfg.min_remaining:
                    is_final_stage = True

            n_rounds = (
                self.cfg.base_cfg.n_rounds * self.cfg.final_round_multiplier
                if is_final_stage
                else self.cfg.base_cfg.n_rounds
            )
            stage_cfg = self._clone_cfg_for_stage(stage_idx, n_rounds)

            stage_df_in = work_df[work_df["identifier"].isin(current_ids)].copy()

            if stage_idx == 1 and self.cfg.rate_first_round:
                rate_cfg = RateConfig(
                    attributes=stage_cfg.attributes,
                    save_dir=stage_cfg.save_dir,
                    file_name=f"stage{stage_idx}_ratings.csv",
                    model=stage_cfg.model,
                    n_parallels=stage_cfg.n_parallels,
                    use_dummy=stage_cfg.use_dummy,
                    modality=stage_cfg.modality,
                    n_attributes_per_run=stage_cfg.n_attributes_per_run,
                    reasoning_effort=stage_cfg.reasoning_effort,
                    reasoning_summary=stage_cfg.reasoning_summary,
                )
                rater = Rate(rate_cfg)
                stage_df_out = await rater.run(
                    stage_df_in, column_name="text", reset_files=reset_files
                )
            else:
                ranker = Rank(stage_cfg)
                stage_df_out = await ranker.run(
                    stage_df_in, column_name="text", reset_files=reset_files
                )

            if self.cfg.keep_stage_columns:
                stage_cols = [c for c in stage_df_out.columns if c != "text"]
                if self.cfg.add_stage_suffix:
                    renamed = {
                        c: f"stage{stage_idx}_{c}" for c in stage_cols if c != "identifier"
                    }
                    stage_df_stage = stage_df_out.rename(columns=renamed)
                else:
                    stage_df_stage = stage_df_out.copy()
                self._stage_dfs.append(stage_df_stage)

            self._update_cumulative(stage_df_out)

            if is_final_stage:
                for i in current_ids:
                    exit_stage[i] = stage_idx
                break

            next_ids = self._select_next_ids(current_ids)
            removed = set(current_ids) - set(next_ids)
            for i in removed:
                exit_stage[i] = stage_idx

            work_df = self._maybe_rewrite_texts(
                df=work_df, ids_to_keep=next_ids, stage_idx=stage_idx, text_col="text"
            )
            current_ids = next_ids

        cum_rows = []
        for i in all_ids:
            row = {"identifier": i}
            for attr in self._attr_list:
                row[f"cumulative_{attr}"] = self._cumulative_scores[attr][i]
            cum_rows.append(row)
        cum_df = pd.DataFrame(cum_rows)

        exit_df = pd.DataFrame(
            {"identifier": list(exit_stage.keys()), "exit_stage": list(exit_stage.values())}
        )

        final_raw = stage_df_out.copy()
        final_raw_cols = [c for c in final_raw.columns if c != "text"]
        final_raw = final_raw.rename(
            columns={c: (c if c == "identifier" else f"final_{c}") for c in final_raw_cols}
        )

        latest_text_df = work_df[["identifier", "text"]].copy()

        out = (
            cum_df.merge(exit_df, on="identifier", how="left")
            .merge(latest_text_df, on="identifier", how="left")
            .merge(final_raw, on="identifier", how="left")
        )
        if self.cfg.keep_stage_columns and self._stage_dfs:
            for sdf in self._stage_dfs:
                out = out.merge(sdf, on="identifier", how="left")

        prefixed_cum = [c for c in out.columns if c.startswith("cumulative_")]
        prefixed_final = [c for c in out.columns if c.startswith("final_")]
        cols = ["identifier", "text", "exit_stage"] + prefixed_cum + prefixed_final
        remaining = [c for c in out.columns if c not in cols]
        out = out[cols + remaining]

        final_path = os.path.join(self._base_folder, "recursive_final.csv")
        out.to_csv(final_path, index=False)
        return out

