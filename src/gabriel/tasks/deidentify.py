from __future__ import annotations

import ast
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils import safest_json
from ..utils.openai_utils import get_all_responses


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class DeidentifyConfig:
    """Configuration for :class:`Deidentifier`."""

    model: str = "gpt-5-mini"
    n_parallels: int = 750
    save_dir: str = "deidentify"
    file_name: str = "deidentified.csv"
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    max_words_per_call: int = 7500
    guidelines: str = ""
    additional_guidelines: str = ""
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    n_passes: int = 1
    use_existing_mappings_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.n_passes, int) or self.n_passes < 1:
            raise ValueError("n_passes must be an integer >= 1")


# ────────────────────────────
# Main de-identification task
# ────────────────────────────
class Deidentifier:
    """Iterative de-identification of sensitive entities in text."""

    def __init__(
        self,
        cfg: DeidentifyConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        if template is not None and template_path is not None:
            raise ValueError("Provide either template or template_path, not both")
        if template_path is not None:
            template = PromptTemplate.from_file(
                template_path, reference_filename="deidentification_prompt.jinja2"
            )
        self.template = template or PromptTemplate.from_package("deidentification_prompt.jinja2")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_by_words(text: str, max_words: int) -> List[str]:
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    @staticmethod
    def _coerce_mapping(value: Any) -> Optional[Dict[str, Any]]:
        """Attempt to convert ``value`` into a mapping dictionary."""

        if value is None:
            return None
        try:
            if pd.isna(value):  # type: ignore[arg-type]
                return None
        except Exception:
            pass

        if isinstance(value, dict):
            return deepcopy(value)

        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned or cleaned.lower() in {"nan", "none"}:
                return None
            try:
                parsed = json.loads(cleaned)
            except Exception:
                try:
                    parsed = ast.literal_eval(cleaned)
                except Exception:
                    return None
            if isinstance(parsed, dict):
                return deepcopy(parsed)
            return None

        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        grouping_column: Optional[str] = None,
        mapping_column: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Deidentify all texts in ``df[column_name]``.

        Parameters
        ----------
        df:
            Input DataFrame.
        column_name:
            Name of the column containing the text to de-identify.
        grouping_column:
            Optional column whose values determine which rows belong to the same
            individual/entity. When omitted, each row is treated independently.
        mapping_column:
            Optional column containing pre-existing mapping dictionaries. The
            first non-empty mapping encountered for each group is used as the
            warm start and is also the mapping reused when
            ``use_existing_mappings_only`` is ``True``.
        reset_files:
            When ``True``, intermediate CSV logs from :func:`get_all_responses`
            are regenerated.
        **kwargs:
            Additional keyword arguments forwarded to
            :func:`gabriel.utils.openai_utils.get_all_responses`.
        """

        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        if grouping_column is not None and grouping_column not in df.columns:
            raise ValueError(
                f"Grouping column '{grouping_column}' not found in DataFrame"
            )
        if mapping_column is not None and mapping_column not in df.columns:
            raise ValueError(
                f"Mapping column '{mapping_column}' not found in DataFrame"
            )

        df_proc = df.reset_index(drop=True).copy()

        if grouping_column is None:
            df_proc["group_id"] = df_proc.index.astype(str)
        else:
            df_proc["group_id"] = df_proc[grouping_column].astype(str)

        group_ids = df_proc["group_id"].unique().tolist()
        base_name = Path(self.cfg.file_name).stem
        csv_path = Path(self.cfg.save_dir) / f"{base_name}_cleaned.csv"
        raw_prefix = Path(self.cfg.save_dir) / f"{base_name}_raw_responses"

        group_segments: Dict[str, List[str]] = {}
        for gid in group_ids:
            segs: List[str] = []
            texts = (
                df_proc.loc[df_proc["group_id"] == gid, column_name]
                .fillna("")
                .astype(str)
                .tolist()
            )
            for text in texts:
                segs.extend(self._chunk_by_words(text, self.cfg.max_words_per_call))
            group_segments[gid] = segs

        group_to_map: Dict[str, dict] = {gid: {} for gid in group_ids}

        warm_start_count = 0
        if mapping_column is not None:
            for gid, subset in df_proc.groupby("group_id"):
                values = subset[mapping_column].tolist()
                mapping = next(
                    (m for m in (self._coerce_mapping(v) for v in values) if m is not None),
                    None,
                )
                if mapping is not None:
                    group_to_map[str(gid)] = mapping
                    warm_start_count += 1
            if warm_start_count:
                print(
                    f"[Deidentify] Loaded {warm_start_count} warm-start mapping(s) "
                    f"from column '{mapping_column}'."
                )
            else:
                print(
                    f"[Deidentify] Column '{mapping_column}' provided but no usable "
                    "mappings were found."
                )

        print(
            "[Deidentify] Tip: edit the first mapping for each group and rerun with "
            "use_existing_mappings_only=True to apply your changes without new LLM calls."
        )

        max_rounds = max(len(segs) for segs in group_segments.values()) if group_segments else 0

        if self.cfg.use_existing_mappings_only:
            print(
                "[Deidentify] use_existing_mappings_only=True -> skipping LLM calls and "
                "reusing provided mappings."
            )
            missing = [gid for gid, mapping in group_to_map.items() if not mapping]
            if missing:
                print(
                    "[Deidentify] Warning: no mapping provided for "
                    f"{len(missing)} group(s). Their text will be returned unchanged."
                )
        else:
            for pass_idx in range(self.cfg.n_passes):
                if self.cfg.n_passes > 1:
                    print(
                        f"[Deidentify] Starting pass {pass_idx + 1}/{self.cfg.n_passes}."
                    )

                for rnd in range(max_rounds):
                    prompts: List[str] = []
                    identifiers: List[str] = []
                    id_to_gid: Dict[str, str] = {}

                    for gid in group_ids:
                        segs = group_segments[gid]
                        if rnd >= len(segs):
                            continue

                        ident = f"{gid}_pass{pass_idx}_seg_{rnd}"
                        identifiers.append(ident)
                        id_to_gid[ident] = gid
                        prompts.append(
                            self.template.render(
                                text=segs[rnd],
                                current_map=json.dumps(
                                    group_to_map.get(gid, {}), ensure_ascii=False
                                ),
                                guidelines=self.cfg.guidelines,
                                additional_guidelines=self.cfg.additional_guidelines,
                            )
                        )

                    if not prompts:
                        continue

                    save_path = raw_prefix.with_name(
                        f"{raw_prefix.name}_pass{pass_idx}_round{rnd}.csv"
                    )
                    batch_df = await get_all_responses(
                        prompts=prompts,
                        identifiers=identifiers,
                        n_parallels=self.cfg.n_parallels,
                        model=self.cfg.model,
                        save_path=str(save_path),
                        use_dummy=self.cfg.use_dummy,
                        max_timeout=self.cfg.max_timeout,
                        json_mode=True,
                        reasoning_effort=self.cfg.reasoning_effort,
                        reasoning_summary=self.cfg.reasoning_summary,
                        reset_files=reset_files,
                        **kwargs,
                    )

                    for ident, resp in zip(batch_df["Identifier"], batch_df["Response"]):
                        gid = id_to_gid.get(ident)
                        if gid is None:
                            continue
                        main = resp[0] if isinstance(resp, list) and resp else ""
                        parsed = await safest_json(main)
                        if isinstance(parsed, dict):
                            group_to_map[gid] = parsed

        mappings_col: List[dict] = []
        deidentified_texts: List[str] = []
        for _, row in df_proc.iterrows():
            gid = row["group_id"]
            mapping = group_to_map.get(gid, {})
            mappings_col.append(mapping)
            text = str(row[column_name])
            deid_text = text
            pairs: List[tuple[str, str]] = []
            for entry in mapping.values():
                if isinstance(entry, dict):
                    casted = entry.get("casted form", "")
                    for real in entry.get("real forms", []) or []:
                        if casted and real:
                            pairs.append((real, casted))
            pairs.sort(key=lambda x: len(x[0]), reverse=True)
            for real, casted in pairs:
                pattern = re.compile(rf"\b{re.escape(real)}\b", flags=re.IGNORECASE)
                deid_text = pattern.sub(casted, deid_text)
            deidentified_texts.append(deid_text)

        df_proc["mapping"] = mappings_col
        df_proc["deidentified_text"] = deidentified_texts
        df_proc.to_csv(csv_path, index=False)
        return df_proc
