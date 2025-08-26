from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


@dataclass
class FilterConfig:
    """Configuration for :class:`Filter`."""

    condition: str
    save_dir: str
    file_name: str = "filter_responses.csv"
    model: str = "gpt-5-nano"
    n_parallels: int = 750
    entities_per_call: int = 150
    shuffle: bool = True
    random_seed: int = 42
    n_runs: int = 1
    threshold: float = 0.5
    additional_instructions: str = ""
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    fix_json_with_llm: bool = False
    json_fix_timeout: Optional[float] = 60.0


class Filter:
    """Filter entities in a DataFrame column based on a condition using an LLM."""

    def __init__(
        self,
        cfg: FilterConfig,
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
                template_path, reference_filename="filter_prompt.jinja2"
            )
        self.template = template or PromptTemplate.from_package("filter_prompt.jinja2")

    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        raw_entities = [str(x) for x in df_proc[column_name].dropna()]

        # unique while preserving order
        seen: set[str] = set()
        entities: List[str] = []
        for ent in raw_entities:
            if ent not in seen:
                seen.add(ent)
                entities.append(ent)

        prompts: List[str] = []
        identifiers: List[str] = []
        for run in range(self.cfg.n_runs):
            ents = list(entities)
            if self.cfg.shuffle:
                rnd = random.Random(self.cfg.random_seed + run)
                rnd.shuffle(ents)
            chunks = [
                ents[i : i + self.cfg.entities_per_call]
                for i in range(0, len(ents), self.cfg.entities_per_call)
            ]
            for idx, chunk in enumerate(chunks):
                prompts.append(
                    self.template.render(
                        condition=self.cfg.condition,
                        entities=chunk,
                        additional_instructions=self.cfg.additional_instructions,
                    )
                )
                identifiers.append(f"filter_{run:03d}_{idx:05d}")

        save_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        if prompts:
            resp_df = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=save_path,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                json_mode=True,
                reset_files=reset_files,
                **kwargs,
            )
        else:
            resp_df = pd.DataFrame(columns=["Identifier", "Response"])

        resp_map: Dict[str, Any] = dict(
            zip(resp_df.get("Identifier", []), resp_df.get("Response", []))
        )

        meets_by_run: List[set[str]] = [set() for _ in range(self.cfg.n_runs)]
        for ident, raw in resp_map.items():
            parts = ident.split("_")
            if len(parts) < 3:
                continue
            try:
                run_idx = int(parts[1])
            except ValueError:
                continue
            parsed = await safest_json(
                raw,
                model=self.cfg.model if self.cfg.fix_json_with_llm else None,
                use_llm_fallback=self.cfg.fix_json_with_llm,
                llm_timeout=self.cfg.json_fix_timeout,
            )
            ent_list: Optional[List[str]] = None
            if isinstance(parsed, dict):
                val = parsed.get("entities meeting condition") or parsed.get(
                    "entities_meeting_condition"
                )
                if isinstance(val, list):
                    ent_list = [str(v) for v in val if isinstance(v, str)]
            elif isinstance(parsed, list):
                ent_list = [str(v) for v in parsed if isinstance(v, str)]
            if ent_list and 0 <= run_idx < self.cfg.n_runs:
                for ent in ent_list:
                    meets_by_run[run_idx].add(ent.strip())

        run_cols: List[str] = []
        for run_idx in range(self.cfg.n_runs):
            meets_norm = {m.lower() for m in meets_by_run[run_idx]}
            col = f"meets_condition_run_{run_idx + 1}"
            run_cols.append(col)
            df_proc[col] = [
                str(v).lower() in meets_norm if not pd.isna(v) else False
                for v in df_proc[column_name]
            ]

        df_proc["meets_condition"] = (
            df_proc[run_cols].sum(axis=1) / self.cfg.n_runs >= self.cfg.threshold
        )
        return df_proc
