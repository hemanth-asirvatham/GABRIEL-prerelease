from __future__ import annotations

import hashlib
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, DefaultDict

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json, load_image_inputs, load_audio_inputs


@dataclass
class ExtractConfig:
    attributes: Dict[str, str]
    save_dir: str = "extraction"
    file_name: str = "extraction.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 400
    n_runs: int = 1
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    additional_instructions: Optional[str] = None
    modality: str = "text"
    n_attributes_per_run: int = 8
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Extract:
    """Extract attributes from passages using an LLM."""

    def __init__(self, cfg: ExtractConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("extraction_prompt.jinja2")

    async def _parse(self, raw: Any, attrs: List[str]) -> Dict[str, str]:
        obj = await safest_json(raw)
        out: Dict[str, str] = {}
        if isinstance(obj, dict):
            for attr in attrs:
                val = obj.get(attr)
                out[attr] = str(val) if val is not None else "unknown"
            return out
        return {attr: "unknown" for attr in attrs}

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        types: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        values = df_proc[column_name].tolist()
        texts = [str(v) for v in values]

        base_ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_val: Dict[str, Any] = {}
        prompt_texts: Dict[str, str] = {}
        row_ids: List[str] = []

        for row, (passage, orig) in enumerate(zip(texts, values)):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            row_ids.append(sha8)
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            id_to_val[sha8] = orig
            prompt_texts[sha8] = passage if self.cfg.modality in {"text", "entity", "web"} else ""
            base_ids.append(sha8)

        df_proc["_gid"] = row_ids

        attr_items = list(self.cfg.attributes.items())
        attr_batches: List[Dict[str, str]] = [
            dict(attr_items[i : i + self.cfg.n_attributes_per_run])
            for i in range(0, len(attr_items), self.cfg.n_attributes_per_run)
        ]

        prompts: List[str] = []
        ids: List[str] = []
        for batch_idx, batch_attrs in enumerate(attr_batches):
            for ident in base_ids:
                prompts.append(
                    self.template.render(
                        text=prompt_texts[ident],
                        attributes=batch_attrs,
                        additional_instructions=self.cfg.additional_instructions or "",
                        modality=self.cfg.modality,
                    )
                )
                ids.append(f"{ident}_batch{batch_idx}")

        prompt_images: Optional[Dict[str, List[str]]] = None
        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None
        if self.cfg.modality == "image":
            tmp: Dict[str, List[str]] = {}
            for ident, rows in id_to_rows.items():
                imgs = load_image_inputs(values[rows[0]])
                if imgs:
                    for batch_idx in range(len(attr_batches)):
                        tmp[f"{ident}_batch{batch_idx}"] = imgs
            prompt_images = tmp or None
        elif self.cfg.modality == "audio":
            tmp_a: Dict[str, List[Dict[str, str]]] = {}
            for ident, rows in id_to_rows.items():
                auds = load_audio_inputs(values[rows[0]])
                if auds:
                    for batch_idx in range(len(attr_batches)):
                        tmp_a[f"{ident}_batch{batch_idx}"] = auds
            prompt_audio = tmp_a or None

        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")

        kwargs.setdefault("use_web_search", self.cfg.modality == "web")

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        if self.cfg.n_runs == 1:
            df_resp_all = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=prompt_images,
                prompt_audio=prompt_audio,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=csv_path,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                json_mode=self.cfg.modality != "audio",
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")
            df_resps = [df_resp_all]
        else:
            prompts_all: List[str] = []
            ids_all: List[str] = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                prompts_all.extend(prompts)
                ids_all.extend([f"{ident}_run{run_idx}" for ident in ids])

            prompt_images_all: Optional[Dict[str, List[str]]] = None
            if prompt_images:
                prompt_images_all = {}
                for ident, imgs in prompt_images.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_images_all[f"{ident}_run{run_idx}"] = imgs
            prompt_audio_all: Optional[Dict[str, List[Dict[str, str]]]] = None
            if prompt_audio:
                prompt_audio_all = {}
                for ident, auds in prompt_audio.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_audio_all[f"{ident}_run{run_idx}"] = auds

            df_resp_all = await get_all_responses(
                prompts=prompts_all,
                identifiers=ids_all,
                prompt_images=prompt_images_all,
                prompt_audio=prompt_audio_all,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=csv_path,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                json_mode=self.cfg.modality != "audio",
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
                df_resps.append(sub)

        full_records: List[Dict[str, Any]] = []
        base_attrs = list(self.cfg.attributes.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_vals: Dict[str, Dict[str, str]] = {
                ident: {attr: "unknown" for attr in base_attrs} for ident in base_ids
            }
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                if "_batch" not in ident_batch:
                    continue
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                batch_idx = int(batch_part)
                attrs = list(attr_batches[batch_idx].keys())
                parsed = await self._parse(raw, attrs)
                for attr in attrs:
                    id_to_vals[base_ident][attr] = parsed.get(attr, "unknown")
            for ident in base_ids:
                parsed = id_to_vals.get(ident, {attr: "unknown" for attr in base_attrs})
                rec = {"id": ident, "text": id_to_val[ident], "run": run_idx}
                rec.update({attr: parsed.get(attr) for attr in base_attrs})
                full_records.append(rec)

        full_df = pd.DataFrame(full_records).set_index(["id", "run"])
        disagg_path = os.path.join(self.cfg.save_dir, f"{base_name}_full_disaggregated.csv")
        full_df.to_csv(disagg_path, index_label=["id", "run"])

        def _pick_first(s: pd.Series) -> str:
            for val in s.dropna():
                if str(val).strip().lower() != "unknown":
                    return str(val)
            return "unknown"

        agg_df = pd.DataFrame({attr: full_df[attr].groupby("id").apply(_pick_first) for attr in base_attrs})

        unknown_counts = {attr: (agg_df[attr] == "unknown").sum() for attr in base_attrs}

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result = df_proc.merge(agg_df, left_on="_gid", right_index=True, how="left")
        result = result.drop(columns=["_gid"])

        result.to_csv(out_path, index=False)

        result = result.replace("unknown", pd.NA)

        if types:
            coerced = result.copy()
            fail_logs: Dict[str, int] = {}
            for col, typ in types.items():
                if col not in coerced:
                    continue
                orig = coerced[col]
                non_null = orig.notna()
                target = str(typ).lower()
                if target in {"datetime", "date"}:
                    conv = pd.to_datetime(orig, errors="coerce")
                else:
                    conv = pd.to_numeric(orig, errors="coerce")
                    if target in {"int", "int64"}:
                        conv = conv.round().astype("Int64")
                coerced[col] = conv
                fail_logs[col] = int((non_null & conv.isna()).sum())
            coerced_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned_coerced.csv")
            coerced.to_csv(coerced_path, index=False)
            for col, n_fail in fail_logs.items():
                print(f"[Extract] Failed to coerce {n_fail} values in column '{col}'.")
            result = coerced

        total = len(agg_df)
        print("\n=== Extraction coverage ===")
        for attr in base_attrs:
            known = total - unknown_counts[attr]
            print(f"{attr:<55s}: {known:5d} extracted, {unknown_counts[attr]:5d} unknown")
        print("============================\n")

        return result
