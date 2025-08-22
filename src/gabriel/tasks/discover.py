from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .codify import Codify
from .compare import Compare, CompareConfig
from .bucket import Bucket, BucketConfig
from .classify import Classify, ClassifyConfig


@dataclass
class DiscoverConfig:
    """Configuration for :class:`Discover`."""

    save_dir: str = "discover"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    n_runs: int = 1
    min_frequency: float = 0.6
    bucket_count: int = 10
    additional_instructions: Optional[str] = None
    differentiate: bool = True
    max_words_per_call: int = 1000
    max_categories_per_call: int = 8
    use_dummy: bool = False
    modality: str = "text"
    n_terms_per_prompt: int = 250
    repeat_bucketing: int = 5
    repeat_voting: int = 20
    next_round_frac: float = 0.5
    top_k_per_round: int = 1
    raw_term_definitions: bool = True
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Discover:
    """High-level feature discovery pipeline.

    Depending on the inputs, the pipeline will either:
    1. Use :class:`Codify` to discover raw feature candidates from a single column, or
    2. Use :class:`Compare` to surface differentiating attributes between two columns.

    The discovered terms are then grouped into buckets via :class:`Bucket` and finally
    applied back onto the dataset using :class:`Classify`.
    """

    def __init__(self, cfg: DiscoverConfig) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg


    async def run(
        self,
        df: pd.DataFrame,
        *,
        column_name: Optional[str] = None,
        circle_column_name: Optional[str] = None,
        square_column_name: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the discovery pipeline.

        Args:
            df: Input dataframe.
            column_name: Column to analyse when using a single column pipeline.
            circle_column_name: First column when contrasting two columns.
            square_column_name: Second column when contrasting two columns.
            reset_files: Forwarded to underlying tasks to control caching.

        Returns:
            Dictionary with intermediate and final results. Keys include:
            ``candidates`` (raw candidate terms), ``buckets`` (bucket definitions),
            ``classification`` (original dataframe with label columns), ``summary`` (if
            circle/square columns were provided) containing per-label differences (``difference_pct``
            expresses circle minus square in percentage points),
            and optionally
            ``compare`` or ``codify`` depending on which stage was used for candidate
            generation.
        """

        single = column_name is not None
        pair = circle_column_name is not None and square_column_name is not None
        if single == pair:
            raise ValueError(
                "Provide either column_name or both circle_column_name and square_column_name"
            )

        compare_df: Optional[pd.DataFrame] = None
        codify_df: Optional[pd.DataFrame] = None

        # ── 1. candidate discovery ─────────────────────────────────────
        if single:
            coder = Codify()
            codify_df = await coder.codify(
                df,
                column_name,  # type: ignore[arg-type]
                categories=None,
                additional_instructions=self.cfg.additional_instructions or "",
                max_words_per_call=self.cfg.max_words_per_call,
                max_categories_per_call=self.cfg.max_categories_per_call,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_dir=os.path.join(self.cfg.save_dir, "codify"),
                reset_files=reset_files,
                debug_print=False,
                use_dummy=self.cfg.use_dummy,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )
            term_defs: Dict[str, str] = {}
            if "coded_passages" in codify_df:
                for entry in codify_df["coded_passages"].dropna():
                    if isinstance(entry, dict):
                        for k, v in entry.items():
                            if k not in term_defs:
                                if isinstance(v, list) and v:
                                    term_defs[k] = str(v[0])
                                else:
                                    term_defs[k] = str(v) if v is not None else ""
            if self.cfg.raw_term_definitions:
                candidate_df = (
                    pd.DataFrame({"term": [term_defs]})
                    if term_defs
                    else pd.DataFrame({"term": []})
                )
            else:
                candidate_df = pd.DataFrame({"term": sorted(set(term_defs.keys()))})
        else:
            cmp_cfg = CompareConfig(
                save_dir=os.path.join(self.cfg.save_dir, "compare"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                max_timeout=None,
                differentiate=self.cfg.differentiate,
                additional_instructions=self.cfg.additional_instructions,
                modality=self.cfg.modality,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )
            cmp = Compare(cmp_cfg)
            compare_df = await cmp.run(
                df,
                circle_column_name,  # type: ignore[arg-type]
                square_column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
            )
            term_defs = {}
            for attr, expl in zip(
                compare_df["attribute"], compare_df["explanation"]
            ):
                if pd.notna(attr) and attr not in term_defs:
                    term_defs[attr] = str(expl) if pd.notna(expl) else ""
            if self.cfg.raw_term_definitions:
                candidate_df = (
                    pd.DataFrame({"term": [term_defs]})
                    if term_defs
                    else pd.DataFrame({"term": []})
                )
            else:
                candidate_df = pd.DataFrame({"term": sorted(set(term_defs.keys()))})

        # ── 2. bucketisation ───────────────────────────────────────────
        bucket_df: pd.DataFrame
        if candidate_df.empty:
            bucket_df = pd.DataFrame(columns=["bucket", "definition"])
        else:
            buck_cfg = BucketConfig(
                bucket_count=self.cfg.bucket_count,
                save_dir=os.path.join(self.cfg.save_dir, "bucket"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                use_dummy=self.cfg.use_dummy,
                additional_instructions=self.cfg.additional_instructions,
                differentiate=self.cfg.differentiate if pair else False,
                n_terms_per_prompt=self.cfg.n_terms_per_prompt,
                repeat_bucketing=self.cfg.repeat_bucketing,
                repeat_voting=self.cfg.repeat_voting,
                next_round_frac=self.cfg.next_round_frac,
                top_k_per_round=self.cfg.top_k_per_round,
                raw_term_definitions=self.cfg.raw_term_definitions,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
            )
            buck = Bucket(buck_cfg)
            bucket_df = await buck.run(
                candidate_df,
                "term",
                reset_files=reset_files,
            )

        labels = (
            dict(zip(bucket_df["bucket"], bucket_df["definition"]))
            if not bucket_df.empty
            else {}
        )

        # ── 3. classification ──────────────────────────────────────────
        classify_result: pd.DataFrame
        summary_df: Optional[pd.DataFrame] = None
        if not labels:
            classify_result = df.reset_index(drop=True).copy()
        elif pair:
            base_cfg = {
                "model": self.cfg.model,
                "n_parallels": self.cfg.n_parallels,
                "n_runs": self.cfg.n_runs,
                "min_frequency": self.cfg.min_frequency,
                "use_dummy": self.cfg.use_dummy,
                "modality": self.cfg.modality,
                "reasoning_effort": self.cfg.reasoning_effort,
                "reasoning_summary": self.cfg.reasoning_summary,
                "n_attributes_per_run": len(labels),
                "differentiate": True,
                "additional_instructions": self.cfg.additional_instructions or "",
            }

            def swap_cs(text: str) -> str:
                def repl(match: re.Match[str]) -> str:
                    word = match.group(0)
                    return "square" if word.lower() == "circle" else "circle"
                return re.sub(r"(?i)circle|square", repl, text)

            circ_cfg = ClassifyConfig(
                labels=labels,
                save_dir=os.path.join(self.cfg.save_dir, "classify_circle"),
                **base_cfg,  # type: ignore[arg-type]
            )

            swapped_labels = {swap_cs(k): swap_cs(v) for k, v in labels.items()}
            sq_cfg = ClassifyConfig(
                labels=swapped_labels,
                save_dir=os.path.join(self.cfg.save_dir, "classify_square"),
                **base_cfg,  # type: ignore[arg-type]
            )

            circ_clf = Classify(circ_cfg)
            sq_clf = Classify(sq_cfg)

            circ_df = await circ_clf.run(
                df,
                circle_column_name=circle_column_name,  # type: ignore[arg-type]
                square_column_name=square_column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
            )
            circ_df = circ_df.rename(columns={lab: f"{lab}_circle" for lab in labels})

            sq_df = await sq_clf.run(
                df,
                circle_column_name=circle_column_name,  # type: ignore[arg-type]
                square_column_name=square_column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
            )
            sq_df = sq_df.rename(columns={swap_cs(lab): f"{lab}_square" for lab in labels})
            sq_cols = [f"{lab}_square" for lab in labels]
            classify_result = circ_df.join(sq_df[sq_cols])
            summary_records: List[Dict[str, Any]] = []
            for lab in labels:
                circ_col = f"{lab}_circle"
                sq_col = f"{lab}_square"
                circ_true = classify_result[circ_col].fillna(False).sum()
                sq_true = classify_result[sq_col].fillna(False).sum()
                total = classify_result[[circ_col, sq_col]].notna().any(axis=1).sum()
                diff = ((circ_true - sq_true) / total * 100) if total else None
                summary_records.append({
                    "label": lab,
                    "circle_true": circ_true,
                    "square_true": sq_true,
                    "total": total,
                "difference_pct": diff,
            })
            summary_df = pd.DataFrame(summary_records)
        else:
            clf_cfg = ClassifyConfig(
                labels=labels,
                save_dir=os.path.join(self.cfg.save_dir, "classify"),
                model=self.cfg.model,
                n_parallels=self.cfg.n_parallels,
                n_runs=self.cfg.n_runs,
                min_frequency=self.cfg.min_frequency,
                additional_instructions=self.cfg.additional_instructions or "",
                use_dummy=self.cfg.use_dummy,
                modality=self.cfg.modality,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                n_attributes_per_run=len(labels),
            )
            clf = Classify(clf_cfg)
            classify_result = await clf.run(
                df,
                column_name,  # type: ignore[arg-type]
                reset_files=reset_files,
            )

        result: Dict[str, Any] = {
            "candidates": candidate_df,
            "buckets": labels,
            "classification": classify_result,
        }
        if not bucket_df.empty:
            result["bucket_df"] = bucket_df
        if summary_df is not None:
            result["summary"] = summary_df
        if compare_df is not None:
            result["compare"] = compare_df
        if codify_df is not None:
            result["codify"] = codify_df
        return result
