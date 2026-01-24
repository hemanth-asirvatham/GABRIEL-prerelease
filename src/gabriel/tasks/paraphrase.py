from __future__ import annotations

import os
import random
import re
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from pathlib import Path

from ..core.prompt_template import PromptTemplate, resolve_template
from ..utils.openai_utils import get_all_responses
from ..utils.logging import announce_prompt_rendering

# Import classifier utilities for recursive validation.  Importing from
# ``gabriel.tasks.classify`` does not introduce a circular dependency
# because the classifier does not import the paraphrasing task.
from ..tasks.classify import Classify, ClassifyConfig


@dataclass
class ParaphraseConfig:
    """Configuration for :class:`Paraphrase`."""

    # Instruction passed to the paraphrase prompt.  This should describe
    # how the original passage should be rewritten.
    instructions: str
    # Optional name for the revised column in the output DataFrame.  If
    # unspecified, ``f"{column_name}_revised"`` will be used.
    revised_column_name: Optional[str] = None
    # Number of revisions to generate per passage.  When greater than 1,
    # additional columns will be appended to the output DataFrame with
    # suffixes ``_1``, ``_2``, etc.
    n_revisions: int = 1
    # Directory where all paraphrase responses and intermediate files are
    # persisted.
    save_dir: str = "paraphrase"
    # Base file name for the raw paraphrase responses.  Cleaned and
    # validated responses will be written using the same stem with
    # ``_cleaned.csv`` appended.
    file_name: str = "paraphrase_responses.csv"
    # OpenAI model used for both paraphrasing and classification.  The
    # default matches the existing behaviour of GABRIEL.
    model: str = "gpt-5-mini"
    # When true, the model will be asked to output JSON only.
    json_mode: bool = False
    # When set, controls whether the underlying helper should use web search
    # augmentation.  ``None`` defers to downstream defaults while ``True`` and
    # ``False`` explicitly enable or disable the feature.
    web_search: Optional[bool] = None
    # Maximum number of parallel requests that will be sent to the
    # underlying API.  Note that classification and paraphrasing share
    # this value for simplicity.
    n_parallels: int = 650
    # Use dummy responses instead of real API calls.  Exposed here
    # primarily for testing.
    use_dummy: bool = False
    # Optional reasoning effort passed through to the LLM helper.
    reasoning_effort: Optional[str] = None
    # Optional reasoning summary passed through to the LLM helper.
    reasoning_summary: Optional[str] = None
    # Maximum number of paraphrase/validation rounds to run.  A value of
    # ``1`` preserves the historical behaviour of a single paraphrase
    # generation pass with no recursive validation.  Values greater than
    # one enable recursive validation with an upper bound on the number of
    # cycles.
    n_rounds: int = 1
    # Deprecated flag kept for backwards compatibility.  When set to
    # ``True`` and ``n_rounds`` is left at its default, it will be coerced
    # to ``2``.
    recursive_validation: Optional[bool] = None
    # When greater than one, multiple paraphrases are generated for
    # each passage in the very first round of generation.  If at least
    # one candidate passes the validation check, that candidate will be
    # selected immediately without triggering further rounds.  A value
    # of one preserves the historical behaviour of producing a single
    # paraphrase per passage at the outset.
    n_initial_candidates: int = 1
    # Number of candidate paraphrases to generate for each failing
    # passage in subsequent validation rounds.  This value is used
    # whenever a paraphrase does not initially satisfy the validation
    # criterion.  Generating multiple candidates in later rounds
    # improves the probability of finding an acceptable paraphrase.
    n_validation_candidates: int = 5
    # Whether to feed the previously chosen paraphrase back into the
    # generator during recursive validation.  When ``False`` (the
    # default), the original text is always used as the source for
    # regeneration.  When ``True``, the most recent paraphrase is
    # provided as the input for further rewriting.  This can be useful
    # when incremental improvements are desired rather than starting
    # over from the original each time.  The option only has effect
    # when ``recursive_validation`` is enabled.
    use_modified_source: bool = False

    def __post_init__(self) -> None:
        try:
            rounds = int(self.n_rounds)
        except (TypeError, ValueError):
            rounds = 1
        if rounds < 1:
            rounds = 1

        if self.recursive_validation is not None:
            warnings.warn(
                "recursive_validation is deprecated; use n_rounds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.recursive_validation and rounds <= 1:
                rounds = 2

        self.n_rounds = rounds


class Paraphrase:
    """Paraphrase text columns in a DataFrame."""

    def __init__(
        self,
        cfg: ParaphraseConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="paraphrase_prompt.jinja2",
        )

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Paraphrase ``df[column_name]`` and return a DataFrame with revisions.

        This method orchestrates prompt construction, asynchronous API
        calls, optional recursive validation using the classifier, and
        persistence of results.  The output DataFrame preserves the
        order of the input and appends one or more revised columns.
        """
        # Ensure row indices are contiguous so we can map responses back
        # deterministically.  A copy is created to avoid mutating the
        # caller's DataFrame.
        df_proc = df.reset_index(drop=True).copy()
        # Convert the target column into a list of strings.  We coerce
        # values to strings so that non-string columns (e.g. numbers) are
        # handled gracefully.
        texts: List[str] = df_proc[column_name].astype(str).tolist()
        # Determine the base name for the revised column(s).
        base_col = self.cfg.revised_column_name or f"{column_name}_revised"
        # Determine how many paraphrases to produce per passage.  A value
        # less than 1 defaults to a single revision to align with
        # existing behaviour.
        n = self.cfg.n_revisions if self.cfg.n_revisions and self.cfg.n_revisions > 0 else 1

        announce_prompt_rendering("Paraphrase", len(texts) * n)

        # Resolve the number of recursive validation rounds to run.  This
        # value is consumed here (and not forwarded to
        # ``get_all_responses``) to avoid unexpected keyword errors.
        requested_rounds = kwargs.pop("n_rounds", self.cfg.n_rounds)
        try:
            max_rounds = int(requested_rounds)
        except (TypeError, ValueError):
            max_rounds = self.cfg.n_rounds
        if max_rounds < 1:
            max_rounds = 1

        # Track whether each revision ultimately received validation
        # approval.  In the non-recursive case every output is treated as
        # approved.
        approval_map: Dict[Tuple[int, int], bool] = {}

        # When recursive validation is disabled (or limited to a single
        # round), follow the original behaviour: generate a single
        # paraphrase per requested revision and skip classification.
        # Otherwise, defer generation and validation to the recursive
        # routine.
        if max_rounds <= 1:
            prompts: List[str] = []
            identifiers: List[str] = []
            for idx, text in enumerate(texts):
                for j in range(1, n + 1):
                    prompts.append(
                        self.template.render(text=text, instructions=self.cfg.instructions)
                    )
                    identifiers.append(f"row_{idx}_rev{j}")
            save_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                save_path=save_path,
                model=self.cfg.model,
                json_mode=self.cfg.json_mode,
                web_search=self.cfg.web_search,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            resp_map: Dict[Tuple[int, int], str] = {}
            for ident, resp in zip(resp_df["Identifier"], resp_df["Response"]):
                main = resp[0] if isinstance(resp, list) and resp else resp
                m = re.match(r"row_(\d+)_rev(\d+)", ident)
                if m:
                    row = int(m.group(1))
                    rev = int(m.group(2)) - 1
                    resp_map[(row, rev)] = main
                    approval_map[(row, rev)] = True
        else:
            # Initialise an empty response map.  The recursive validation
            # routine will populate this map with one paraphrase per
            # (row, revision) key and record whether it passed
            # validation.
            resp_map: Dict[Tuple[int, int], str] = {}
            await self._recursive_validate(
                texts,
                resp_map,
                approval_map,
                reset_files=reset_files,
                max_rounds=max_rounds,
            )

        # Assemble the final columns.  When multiple revisions are
        # requested, each revision will occupy its own column with a
        # numeric suffix.
        col_names = [base_col] if n == 1 else [f"{base_col}_{i}" for i in range(1, n + 1)]
        approval_cols = (
            [f"{base_col}_approved"]
            if n == 1
            else [f"{col}_approved" for col in col_names]
        )
        for j, col in enumerate(col_names):
            df_proc[col] = [resp_map.get((i, j), "") for i in range(len(df_proc))]
            df_proc[approval_cols[j]] = [
                bool(approval_map.get((i, j), True)) for i in range(len(df_proc))
            ]

        # Persist the cleaned and validated DataFrame to disk.  This file
        # excludes metadata columns such as ``Identifier`` or ``Response``
        # and is intended for downstream analysis.
        out_path = os.path.join(
            self.cfg.save_dir,
            f"{os.path.splitext(self.cfg.file_name)[0]}_cleaned.csv",
        )
        df_proc.to_csv(out_path, index=False)
        return df_proc

    async def _recursive_validate(
        self,
        original_texts: List[str],
        resp_map: Dict[Tuple[int, int], str],
        approval_map: Dict[Tuple[int, int], bool],
        *,
        reset_files: bool = False,
        max_rounds: int,
    ) -> None:
        """
        Generate and validate paraphrases for each passage using a
        classifier.  This routine unifies initial and subsequent
        candidate generation by allowing a configurable number of
        candidates on the first round (``n_initial_candidates``) and a
        separate number for later rounds (``n_validation_candidates``).
        Candidates that pass validation are accepted immediately.  For
        candidates that fail, further paraphrases are generated until
        either a valid paraphrase is found or no new paraphrases can
        be produced.

        If ``use_modified_source`` is ``True``, subsequent rounds will
        generate new paraphrases from the most recently chosen
        paraphrase rather than starting from the original text.  In
        either case, the classifier always evaluates the modified
        candidate against the original text to ensure the original
        instruction has been followed.
        """
        # Determine the number of revisions (columns) to produce.  At
        # least one revision is always generated.  This mirrors the logic
        # in :meth:`run`.
        n_revs = self.cfg.n_revisions if self.cfg.n_revisions and self.cfg.n_revisions > 0 else 1
        # Build a list of keys for every passage/revision pair.  Keys
        # encode the row index and zero-based revision index.
        all_keys: List[Tuple[int, int]] = [
            (row_idx, rev_idx)
            for row_idx in range(len(original_texts))
            for rev_idx in range(n_revs)
        ]

        # We'll use this list to track which keys still require
        # validation in each round.  Initially, all keys are awaiting
        # generation and validation.
        to_check: List[Tuple[int, int]] = list(all_keys)
        round_number = 0

        # Create the classifier configuration once.  A dedicated
        # validation directory is used to store classification results.
        validation_dir = os.path.join(self.cfg.save_dir, "validation")
        os.makedirs(validation_dir, exist_ok=True)
        # A single label is used to indicate whether the instructions
        # were followed.  The definition is intentionally phrased in a
        # slightly more permissive way than before to reduce the false
        # rejection rate.
        labels = {
            "instructions_followed": (
                "Return True if the instructions were largely (even if not perfectly) followed in turning the "
                "original text into the modified text (i.e. the modified text mostly exhibits the spirit of the instructions "
                "even if not everything is exact). Be quite forgiving; understand that the modifications won't be perfect. "
                "Ensure the spirit of the instructions is followed, even if not word for word. "
                "False otherwise, if there are still important shortcomings in the modified text vis a vis the instructions."
            )
        }
        classify_cfg = ClassifyConfig(
            labels=labels,
            save_dir=validation_dir,
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        classifier = Classify(classify_cfg)

        # Continue looping until there are no passages left to validate
        # or until we hit the round limit.
        last_candidate_map: Dict[Tuple[int, int], List[str]] = {}
        while to_check and round_number < max_rounds:
            # Determine how many candidate paraphrases to generate per
            # passage for this round.  The first round uses
            # ``n_initial_candidates``; later rounds use
            # ``n_validation_candidates``.
            if round_number == 0:
                candidates_per_key = max(self.cfg.n_initial_candidates, 1)
            else:
                candidates_per_key = max(self.cfg.n_validation_candidates * round_number, 1)

            # Build paraphrase prompts for every key still requiring
            # validation.  Each key may produce multiple candidates.
            prompts: List[str] = []
            identifiers: List[str] = []
            for key in to_check:
                row_idx, rev_idx = key
                # Choose the base text according to whether we reuse
                # modified text in later rounds.  On the first round we
                # always use the original text.  On subsequent rounds,
                # if ``use_modified_source`` is true and a paraphrase
                # exists for this key, use that paraphrase as the base
                # for regeneration.  Otherwise, continue to use the
                # original.
                if round_number > 0 and self.cfg.use_modified_source and key in resp_map:
                    base_text = resp_map[key]
                else:
                    base_text = original_texts[row_idx]
                for cand_idx in range(candidates_per_key):
                    prompts.append(
                        self.template.render(text=base_text, instructions=self.cfg.instructions)
                    )
                    # Encode row, revision, round and candidate index in
                    # the identifier.  Revision numbers are stored one-
                    # based in the identifier for backwards compatibility.
                    identifiers.append(
                        f"row_{row_idx}_rev{rev_idx + 1}_round{round_number}_cand{cand_idx}"
                    )

            # If no prompts were constructed (which may happen if
            # ``candidates_per_key`` is zero), break to avoid an
            # infinite loop.
            if not prompts:
                break

            announce_prompt_rendering("Paraphrase:validate", len(prompts))

            # Write the prompts to the paraphrasing API.  We construct a
            # unique filename for each round to preserve intermediate
            # results.  The responses are appended to any existing
            # files unless ``reset_files`` is true.
            tmp_save_path = os.path.join(
                self.cfg.save_dir,
                f"{os.path.splitext(self.cfg.file_name)[0]}_round{round_number}.csv",
            )
            new_resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                save_path=tmp_save_path,
                model=self.cfg.model,
                json_mode=self.cfg.json_mode,
                web_search=self.cfg.web_search,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )

            # Organise API responses by (row, revision) so that
            # classification can be performed in bulk.  Each key
            # corresponds to a list of candidate paraphrases.
            candidate_map: Dict[Tuple[int, int], List[str]] = {
                key: [] for key in to_check
            }
            for ident, resp in zip(new_resp_df["Identifier"], new_resp_df["Response"]):
                text = resp[0] if isinstance(resp, list) and resp else resp
                # Parse the identifier back into row and revision indices.
                m = re.match(r"row_(\d+)_rev(\d+)_round\d+_cand(\d+)", ident)
                if m:
                    row_i = int(m.group(1))
                    rev_i = int(m.group(2)) - 1
                    candidate_map.setdefault((row_i, rev_i), []).append(text)

            last_candidate_map = candidate_map

            # Build classification prompts for every candidate across all
            # keys.  We keep a parallel list of (row, revision, candidate
            # index) so we can map results back after classification.
            cand_prompts: List[str] = []
            cand_keys: List[Tuple[int, int, int]] = []
            for key in to_check:
                row_idx, rev_idx = key
                orig_text = original_texts[row_idx]
                candidates = candidate_map.get(key, [])
                for cand_index, cand_text in enumerate(candidates):
                    cand_text = cand_text or ""
                    cls_prompt = (
                        "INSTRUCTIONS:\n"
                        f"{self.cfg.instructions.strip()}\n\n"
                        "BEGIN ORIGINAL TEXT:\n"
                        f"{orig_text.strip()}\n"
                        "END ORIGINAL TEXT\n\n"
                        "BEGIN MODIFIED TEXT:\n"
                        f"{cand_text.strip()}\n"
                        "END MODIFIED TEXT\n\n"
                        "Previously, the original text was taken and modified, following the provided instructions, to create the modified text. "
                        "Does the modified text faithfully apply the instructions as a transformation of "
                        "the original text? Answer True if the modification follows "
                        "the instructions to a satisfactory, though not necessarily perfect, degree. "
                        "Tolerate some imperfection and inconsistency, as long as the spirit of the instructions is obeyed to the extent that is reasonable. Be reasonably forgiving. "
                        "Again, the modification instructions that need to be validated are: "
                        f"{self.cfg.instructions.strip()}\n\n"
                    )
                    cand_prompts.append(cls_prompt)
                    cand_keys.append((row_idx, rev_idx, cand_index))

            # Run the classifier on all candidate prompts.  If there are
            # no candidates (which should not occur), produce an empty
            # DataFrame to avoid indexing errors.
            if cand_prompts:
                cand_df = pd.DataFrame({"text": cand_prompts})
                cand_res_df = await classifier.run(
                    cand_df, column_name="text", reset_files=reset_files
                )
            else:
                cand_res_df = pd.DataFrame()

            # Build a lookup from (row, revision) to a list of boolean
            # classification results corresponding to each candidate.  A
            # candidate passes if the classifier returns True or None.  A
            # None value indicates uncertainty but is treated as a pass
            # here to reduce the false rejection rate.
            cand_results_map: Dict[Tuple[int, int], List[bool]] = {
                key: [False] * len(candidate_map.get(key, [])) for key in to_check
            }
            for idx, (row_idx, rev_idx, cand_index) in enumerate(cand_keys):
                if not cand_res_df.empty:
                    flag = cand_res_df.loc[idx, "instructions_followed"]
                else:
                    flag = None
                # Treat None (uncertain) as a pass and only count False
                # values as failures.
                cand_results_map[(row_idx, rev_idx)][cand_index] = bool(flag) or flag is None

            # Determine which passages still require another round and
            # select the best candidate for each key.  For each key, the
            # first passing candidate is chosen.  If no candidate passed,
            # the first candidate is chosen as a fallback and the key
            # remains scheduled for another round.
            next_to_check: List[Tuple[int, int]] = []
            for key in to_check:
                row_idx, rev_idx = key
                candidates = candidate_map.get(key, [])
                results = cand_results_map.get(key, [])
                chosen_text: Optional[str] = None
                passed_flag = False
                for cand_text, passed in zip(candidates, results):
                    if passed:
                        chosen_text = cand_text
                        passed_flag = True
                        break
                # If no candidate passed and at least one candidate
                # exists, choose the first candidate as a fallback.
                if chosen_text is None and candidates:
                    chosen_text = candidates[0]
                    passed_flag = False
                # Update the response map with the chosen paraphrase.
                if chosen_text is not None:
                    resp_map[(row_idx, rev_idx)] = chosen_text
                    approval_map[(row_idx, rev_idx)] = passed_flag
                    # If the candidate did not pass validation, schedule
                    # another round.
                    if not passed_flag:
                        next_to_check.append(key)
                else:
                    # If no candidates were produced (which should not
                    # happen), keep the key for another round to avoid
                    # losing the entry entirely.
                    next_to_check.append(key)

            # Prepare for the next round.
            to_check = next_to_check
            round_number += 1

        # If we exited because we hit the round limit, ensure all
        # remaining keys are assigned a paraphrase and marked as not
        # approved.  When candidates were produced in the last round, we
        # randomly choose one of them as the final fallback.
        if to_check:
            for key in to_check:
                candidates = last_candidate_map.get(key, [])
                if candidates:
                    resp_map[key] = random.choice(candidates)
                approval_map[key] = False
