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
from ..utils import safest_json, safe_json, get_all_embeddings


@dataclass
class MergeConfig:
    """Configuration options for :class:`Merge`."""

    save_dir: str = "merge"
    file_name: str = "merge_responses.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    n_runs: int = 1
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    additional_instructions: Optional[str] = None
    use_embeddings: bool = True
    short_list_len: int = 25
    long_list_len: int = 500


class Merge:
    """Fuzzy merge between two DataFrames using LLM assistance."""

    def __init__(self, cfg: MergeConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("merge_prompt.jinja2")

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(val: str) -> str:
        """Normalize strings for fuzzy matching."""
        return re.sub(r"[^0-9a-z]+", "", val.lower())

    @classmethod
    def _deduplicate(
        cls, series: pd.Series
    ) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
        """Return (unique_values, rep_to_group, norm_to_rep) for a Series."""
        norm_map: Dict[str, str] = {}
        groups: Dict[str, List[str]] = {}
        for val in series.dropna().astype(str):
            norm = cls._normalize(val)
            if norm in norm_map:
                rep = norm_map[norm]
                groups[rep].append(val)
            else:
                norm_map[norm] = val
                groups[val] = [val]
        uniques = list(groups.keys())
        return uniques, groups, norm_map

    # ------------------------------------------------------------------
    async def run(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        *,
        on: Optional[str] = None,
        left_on: Optional[str] = None,
        right_on: Optional[str] = None,
        how: str = "left",
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Perform an LLM-assisted merge between two DataFrames."""

        if on:
            left_key = right_key = on
        elif left_on and right_on:
            left_key, right_key = left_on, right_on
        else:
            raise ValueError("Specify `on` or both `left_on` and `right_on`.")

        how = how.lower()
        if how not in {"left", "right"}:
            raise ValueError("`how` must be either 'left' or 'right'.")

        if how == "left":
            short_df, long_df = df_left.reset_index(drop=True), df_right.reset_index(drop=True)
            short_key, long_key = left_key, right_key
        else:  # right merge
            short_df, long_df = df_right.reset_index(drop=True), df_left.reset_index(drop=True)
            short_key, long_key = right_key, left_key

        # Deduplicate keys and track normalized maps
        short_uniques, short_groups, short_norm_map = self._deduplicate(short_df[short_key])
        long_uniques, long_groups, long_norm_map = self._deduplicate(long_df[long_key])

        use_embeddings = self.cfg.use_embeddings and len(long_uniques) >= self.cfg.long_list_len

        clusters: List[List[str]] = []
        candidates: List[List[str]] = []

        if use_embeddings:
            # Compute embeddings for both unique sets
            short_emb = await get_all_embeddings(
                texts=short_uniques,
                identifiers=short_uniques,
                save_path=os.path.join(self.cfg.save_dir, "short_embeddings.pkl"),
                reset_file=reset_files,
                use_dummy=self.cfg.use_dummy,
                verbose=False,
            )
            long_emb = await get_all_embeddings(
                texts=long_uniques,
                identifiers=long_uniques,
                save_path=os.path.join(self.cfg.save_dir, "long_embeddings.pkl"),
                reset_file=reset_files,
                use_dummy=self.cfg.use_dummy,
                verbose=False,
            )

            if short_emb:
                arr = np.array([short_emb[s] for s in short_uniques], dtype=float)
                k = max(1, int(np.ceil(len(short_uniques) / self.cfg.short_list_len)))
                centroids, labels = kmeans2(arr, k, minit="points")
                centroid_list: List[np.ndarray] = []
                for cluster_id in range(k):
                    members = [short_uniques[i] for i, lbl in enumerate(labels) if lbl == cluster_id]
                    if not members:
                        continue
                    for j in range(0, len(members), self.cfg.short_list_len):
                        subset = members[j : j + self.cfg.short_list_len]
                        clusters.append(subset)
                        centroid_list.append(np.mean([short_emb[m] for m in subset], axis=0))

                long_matrix = np.array([long_emb[t] for t in long_uniques], dtype=float)
                long_norms = np.linalg.norm(long_matrix, axis=1) + 1e-8
                for cent in centroid_list:
                    sims = long_matrix @ cent / (np.linalg.norm(cent) * long_norms)
                    top_idx = np.argsort(sims)[::-1][: self.cfg.long_list_len]
                    candidates.append([long_uniques[i] for i in top_idx])
            else:
                clusters = []
                candidates = []
        else:
            short_sorted = sorted(short_uniques, key=lambda x: x.lower())
            long_sorted = sorted(long_uniques, key=lambda x: x.lower())
            for i in range(0, len(short_sorted), self.cfg.short_list_len):
                clusters.append(short_sorted[i : i + self.cfg.short_list_len])
            if len(long_sorted) <= self.cfg.long_list_len:
                candidates = [long_sorted for _ in clusters]
            else:
                import bisect

                lower_long = [s.lower() for s in long_sorted]
                for clus in clusters:
                    mid = clus[len(clus) // 2].lower()
                    idx = bisect.bisect_left(lower_long, mid)
                    start = max(0, idx - self.cfg.long_list_len // 2)
                    end = start + self.cfg.long_list_len
                    if end > len(long_sorted):
                        end = len(long_sorted)
                        start = max(0, end - self.cfg.long_list_len)
                    candidates.append(long_sorted[start:end])

        prompts: List[str] = []
        identifiers: List[str] = []
        for idx, (short_terms, long_terms) in enumerate(zip(clusters, candidates)):
            short_dict = {s: "" for s in short_terms}
            long_text = "\n".join(long_terms)
            prompts.append(
                self.template.render(
                    short_list=short_dict,
                    long_list=long_text,
                    additional_instructions=self.cfg.additional_instructions or "",
                )
            )
            identifiers.append(f"merge_{idx:05d}")

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

        matches: Dict[str, str] = {}
        for clus, res in zip(clusters, parsed):
            clus_norm = {self._normalize(s): s for s in clus}
            # Handle common cases where the model wraps the dict in a list or
            # returns it as a JSON-encoded string.
            if isinstance(res, list):
                combined: Dict[str, str] = {}
                for item in res:
                    if isinstance(item, dict):
                        combined.update(item)
                    elif isinstance(item, str):
                        inner = safe_json(item)
                        if isinstance(inner, dict):
                            combined.update(inner)
                res = combined
            elif isinstance(res, str):
                res = safe_json(res)

            if isinstance(res, dict):
                for k, v in res.items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        continue
                    k_norm = self._normalize(k)
                    v_norm = self._normalize(v)
                    if k_norm in clus_norm and v_norm in long_norm_map:
                        short_rep = clus_norm[k_norm]
                        long_rep = long_norm_map[v_norm]
                        matches[short_rep] = long_rep

        records: List[Dict[str, str]] = []
        if short_key == long_key:
            temp_col = f"{long_key}_match"
            for short_rep, long_rep in matches.items():
                for s in short_groups.get(short_rep, []):
                    for l in long_groups.get(long_rep, []):
                        records.append({short_key: s, temp_col: l})
            map_df = pd.DataFrame(records, columns=[short_key, temp_col])
            map_df[short_key] = map_df[short_key].astype(object)
            map_df[temp_col] = map_df[temp_col].astype(object)
            merged = short_df.merge(map_df, how="left", on=short_key)
            merged = merged.merge(
                long_df,
                how="left",
                left_on=temp_col,
                right_on=long_key,
                suffixes=("", "_y"),
            )
            merged = merged.drop(columns=[temp_col])
        else:
            for short_rep, long_rep in matches.items():
                for s in short_groups.get(short_rep, []):
                    for l in long_groups.get(long_rep, []):
                        records.append({short_key: s, long_key: l})
            map_df = pd.DataFrame(records, columns=[short_key, long_key])
            map_df[short_key] = map_df[short_key].astype(object)
            map_df[long_key] = map_df[long_key].astype(object)
            merged = short_df.merge(map_df, how="left", on=short_key)
            merged = merged.merge(
                long_df,
                how="left",
                left_on=long_key,
                right_on=long_key,
                suffixes=("", "_y"),
            )
        return merged
