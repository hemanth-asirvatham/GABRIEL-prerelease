from __future__ import annotations

import asyncio
import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate, resolve_template
from ..utils.openai_utils import get_all_responses
from ..utils import (
    safest_json,
    load_image_inputs,
    load_audio_inputs,
)


@dataclass
class CompareConfig:
    save_dir: str = "comparison"
    file_name: str = "comparison_responses.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    n_runs: int = 1
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    differentiate: bool = True
    additional_instructions: Optional[str] = None
    modality: str = "text"
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    circle_first: Optional[bool] = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Compare:
    """Compare two columns row-wise using an LLM."""

    def __init__(
        self,
        cfg: CompareConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="comparison_prompt.jinja2",
        )

    async def _parse(self, raw: Any) -> Dict[str, str]:
        obj = await safest_json(raw)
        if isinstance(obj, dict):
            return {str(k): str(v) if v is not None else "" for k, v in obj.items()}
        return {}

    async def run(
        self,
        df: pd.DataFrame,
        circle_column_name: str,
        square_column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        mask = df_proc[circle_column_name].notna() & df_proc[square_column_name].notna()
        skipped = int((~mask).sum())
        if skipped:
            print(
                f"Skipping {skipped} rows with NaN in {circle_column_name} or {square_column_name}"
            )
        df_proc = df_proc[mask].reset_index(drop=True)

        circles = df_proc[circle_column_name].tolist()
        squares = df_proc[square_column_name].tolist()
        pairs = list(zip(circles, squares))

        prompts: List[str] = []
        ids: List[str] = []
        id_to_circle_first: Dict[str, bool] = {}
        for circle, square in pairs:
            ident = hashlib.sha1(f"{circle}|{square}".encode()).hexdigest()[:8]
            ids.append(ident)
            circle_first_flag = (
                self.cfg.circle_first
                if self.cfg.circle_first is not None
                else random.random() < 0.5
            )
            id_to_circle_first[ident] = circle_first_flag
            circle_text = (
                circle if self.cfg.modality in {"text", "entity", "web"} else ""
            )
            square_text = (
                square if self.cfg.modality in {"text", "entity", "web"} else ""
            )
            prompts.append(
                self.template.render(
                    entry_circle=circle_text,
                    entry_square=square_text,
                    differentiate=self.cfg.differentiate,
                    additional_instructions=self.cfg.additional_instructions or "",
                    modality=self.cfg.modality,
                    circle_first=circle_first_flag,
                )
            )

        prompt_images: Optional[Dict[str, List[str]]] = None
        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None
        if self.cfg.modality == "image":
            tmp: Dict[str, List[str]] = {}
            for ident, (circle, square) in zip(ids, pairs):
                imgs: List[str] = []
                circle_imgs = load_image_inputs(circle)
                square_imgs = load_image_inputs(square)
                if id_to_circle_first.get(ident, False):
                    if circle_imgs:
                        imgs.extend(circle_imgs)
                    if square_imgs:
                        imgs.extend(square_imgs)
                else:
                    if square_imgs:
                        imgs.extend(square_imgs)
                    if circle_imgs:
                        imgs.extend(circle_imgs)
                if imgs:
                    tmp[ident] = imgs
            prompt_images = tmp or None
        elif self.cfg.modality == "audio":
            tmp_a: Dict[str, List[Dict[str, str]]] = {}
            for ident, (circle, square) in zip(ids, pairs):
                auds: List[Dict[str, str]] = []
                circle_auds = load_audio_inputs(circle)
                square_auds = load_audio_inputs(square)
                if id_to_circle_first.get(ident, False):
                    if circle_auds:
                        auds.extend(circle_auds)
                    if square_auds:
                        auds.extend(square_auds)
                else:
                    if square_auds:
                        auds.extend(square_auds)
                    if circle_auds:
                        auds.extend(circle_auds)
                if auds:
                    tmp_a[ident] = auds
            prompt_audio = tmp_a or None

        csv_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)

        kwargs.setdefault("web_search", self.cfg.modality == "web")

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

        resp_map = dict(zip(df_resp_all.Identifier, df_resp_all.Response))
        parsed = await asyncio.gather(*[self._parse(resp_map.get(i, "")) for i in ids])

        records: List[Dict[str, str]] = []
        for (circle, square), res in zip(pairs, parsed):
            for attr, expl in res.items():
                records.append(
                    {
                        circle_column_name: circle,
                        square_column_name: square,
                        "attribute": attr,
                        "explanation": expl,
                    }
                )

        out_df = pd.DataFrame(records)
        if not out_df.empty:
            out_df.set_index([circle_column_name, square_column_name], inplace=True)
        return out_df
