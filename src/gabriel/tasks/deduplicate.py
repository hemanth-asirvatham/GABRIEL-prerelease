from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json, get_all_embeddings


@dataclass
class DeduplicateConfig:
    """Configuration for :class:`Deduplicate`."""

    save_dir: str = "deduplicate"
    file_name: str = "deduplicate_responses.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    n_runs: int = 1
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    additional_instructions: Optional[str] = None
    use_embeddings: bool = True
    group_size: int = 500


class Deduplicate:
    """LLM-assisted deduplication for a single DataFrame column."""

    def __init__(self, cfg: DeduplicateConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("deduplicate_prompt.jinja2")

    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate(series: pd.Series) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
        """Return (unique_values, rep_to_group, orig_to_rep) for a Series."""
        rep_map: Dict[str, str] = {}
        groups: Dict[str, List[str]] = {}
        orig_to_rep: Dict[str, str] = {}
        for val in series.dropna().astype(str):
            norm = re.sub(r"[^0-9a-z]+", "", val.lower())
            if norm in rep_map:
                rep = rep_map[norm]
                groups[rep].append(val)
            else:
                rep_map[norm] = val
                groups[val] = [val]
            orig_to_rep[val] = rep_map[norm]
        uniques = list(groups.keys())
        return uniques, groups, orig_to_rep

    # ------------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        *,
        on: str,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Deduplicate a column using LLM assistance."""

        df_proc = df.reset_index(drop=True).copy()
        uniques, groups, orig_to_rep = self._deduplicate(df_proc[on])

        use_embeddings = self.cfg.use_embeddings and len(uniques) >= self.cfg.group_size

        batches: List[List[str]] = []
        if use_embeddings:
            emb = await get_all_embeddings(
                texts=uniques,
                identifiers=uniques,
                save_path=os.path.join(self.cfg.save_dir, "deduplicate_embeddings.pkl"),
                reset_file=reset_files,
                use_dummy=self.cfg.use_dummy,
                verbose=False,
            )
            if emb:
                arr = np.array([emb[u] for u in uniques], dtype=float)
                k = max(1, int(np.ceil(len(uniques) / self.cfg.group_size)))
                _, labels = kmeans2(arr, k, minit="points")
                for cluster_id in range(k):
                    members = [uniques[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
                    if not members:
                        continue
                    for j in range(0, len(members), self.cfg.group_size):
                        batches.append(members[j : j + self.cfg.group_size])
        else:
            sorted_uniques = sorted(uniques, key=lambda x: x.lower())
            for i in range(0, len(sorted_uniques), self.cfg.group_size):
                batches.append(sorted_uniques[i : i + self.cfg.group_size])

        prompts: List[str] = []
        identifiers: List[str] = []
        for idx, items in enumerate(batches):
            item_text = "\n".join(items)
            prompts.append(
                self.template.render(
                    group_id=f"deduplicate_{idx:05d}",
                    items=item_text,
                    additional_instructions=self.cfg.additional_instructions or "",
                )
            )
            identifiers.append(f"deduplicate_{idx:05d}")

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

        resp_map = dict(zip(resp_df.get("Identifier", []), resp_df.get("Response", [])))
        parsed = await asyncio.gather(
            *[safest_json(resp_map.get(i, "")) for i in identifiers]
        )

        mappings: Dict[str, str] = {}
        for items, res in zip(batches, parsed):
            if isinstance(res, list):
                for row in res:
                    if not isinstance(row, dict):
                        continue
                    inp = row.get("input")
                    mapped = row.get("mapped")
                    if isinstance(inp, str) and isinstance(mapped, str) and inp in items:
                        mappings[inp] = mapped

        for rep in uniques:
            mappings.setdefault(rep, rep)

        mapped_vals: List[Optional[str]] = []
        for val in df_proc[on]:
            if pd.isna(val):
                mapped_vals.append(val)
            else:
                rep = orig_to_rep.get(str(val), str(val))
                mapped_vals.append(mappings.get(rep, rep))
        df_proc[f"mapped_{on}"] = mapped_vals
        return df_proc
