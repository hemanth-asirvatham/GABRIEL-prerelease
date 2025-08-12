# src/gabriel/tasks/rate.py
# ════════════════════════════════════════════════════════════════════
# Robust passage-rating task with optional debug logging.
# ════════════════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional
import os
from pathlib import Path

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class RateConfig:
    attributes: Dict[str, str]
    save_dir: str = "ratings"
    file_name: str = "ratings.csv"
    model: str = "o4-mini"
    n_parallels: int = 400
    n_runs: int = 1
    use_dummy: bool = False
    timeout: float = 60.0
    rating_scale: Optional[str] = None
    additional_instructions: Optional[str] = None
    modality: str = "text"


# ────────────────────────────
# Main rating task
# ────────────────────────────
class Rate:
    """Rate passages on specified attributes (0–100)."""


    # -----------------------------------------------------------------
    def __init__(self, cfg: RateConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("ratings_prompt.jinja2")

    # -----------------------------------------------------------------
    # Parse raw LLM output into {attribute: float}
    # -----------------------------------------------------------------
    async def _parse(self, raw: Any) -> Dict[str, Optional[float]]:
        obj = await safest_json(raw)
        out: Dict[str, Optional[float]] = {}
        if isinstance(obj, dict):
            for attr in self.cfg.attributes:
                try:
                    out[attr] = float(obj.get(attr)) if obj.get(attr) is not None else None
                except Exception:
                    out[attr] = None
            return out

        return {attr: None for attr in self.cfg.attributes}

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        text_column: str,
        *,
        debug: bool = False,
        reset_files: bool = False,
        image_column: Optional[str] = None,
        audio_column: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Return ``df`` with one column per attribute rating."""

        df_proc = df.reset_index(drop=True).copy()
        texts = df_proc[text_column].astype(str).tolist()

        prompts: List[str] = []
        ids: List[str] = []
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_text: Dict[str, str] = {}

        # Build prompts, deduplicating identical passages
        for row, passage in enumerate(texts):
            sha8 = hashlib.sha1(passage.encode()).hexdigest()[:8]
            id_to_rows[sha8].append(row)
            if len(id_to_rows[sha8]) > 1:
                continue
            id_to_text[sha8] = passage
            prompts.append(
                self.template.render(
                    text=passage,
                    attributes=self.cfg.attributes,
                    scale=self.cfg.rating_scale,
                    additional_instructions=self.cfg.additional_instructions,
                    modality=self.cfg.modality,
                )
            )
            ids.append(sha8)

        prompt_images: Optional[Dict[str, List[str]]] = None
        if image_column is not None and image_column in df_proc:
            prompt_images = {}
            img_list = df_proc[image_column].tolist()
            for ident, rows in id_to_rows.items():
                imgs = img_list[rows[0]]
                if imgs:
                    prompt_images[ident] = imgs

        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None
        if audio_column is not None and audio_column in df_proc:
            prompt_audio = {}
            aud_list = df_proc[audio_column].tolist()
            for ident, rows in id_to_rows.items():
                auds = aud_list[rows[0]]
                if auds:
                    prompt_audio[ident] = auds

        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")

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
                timeout=self.cfg.timeout,
                json_mode=True,
                reset_files=reset_files,
                **kwargs,
            )
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
                timeout=self.cfg.timeout,
                json_mode=True,
                reset_files=reset_files,
                **kwargs,
            )

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(suffix + "$", "", regex=True)
                df_resps.append(sub)

        if debug:
            print("\n── raw LLM responses ──")
            for run_idx, df_resp in enumerate(df_resps, start=1):
                for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                    r = raw[0] if isinstance(raw, list) and raw else raw
                    print(f"[run {run_idx}] {ident} →\n{r}\n")
            print("────────────────────────\n")

        # parse each run and build disaggregated records
        full_records: List[Dict[str, Any]] = []
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_ratings: Dict[str, Dict[str, Optional[float]]] = {}
            for ident, raw in zip(df_resp.Identifier, df_resp.Response):
                main = raw[0] if isinstance(raw, list) and raw else raw
                id_to_ratings[ident] = await self._parse(main)
            for ident in ids:
                parsed = id_to_ratings.get(ident, {attr: None for attr in self.cfg.attributes})
                rec = {"text": id_to_text[ident], "run": run_idx}
                rec.update({attr: parsed.get(attr) for attr in self.cfg.attributes})
                full_records.append(rec)

        full_df = pd.DataFrame(full_records).set_index(["text", "run"])
        disagg_path = os.path.join(self.cfg.save_dir, f"{base_name}_full_disaggregated.csv")
        full_df.to_csv(disagg_path, index_label=["text", "run"])

        # aggregate across runs
        agg_df = full_df.groupby("text")[list(self.cfg.attributes)].mean()

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        result = df_proc.merge(agg_df, left_on=text_column, right_index=True, how="left")
        result.to_csv(out_path, index=False)

        # keep raw response files for reference

        return result
