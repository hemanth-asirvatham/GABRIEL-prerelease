from __future__ import annotations

import asyncio
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json


@dataclass
class BucketConfig:
    """Configuration for :class:`Bucket`."""

    bucket_count: int = 10
    save_dir: str = "buckets"
    file_name: str = "bucket_definitions.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 400
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    additional_instructions: Optional[str] = None
    differentiate: bool = False
    n_terms_per_prompt: int = 250
    repeat_bucketing: int = 5
    repeat_voting: int = 20
    next_round_frac: float = 0.5
    top_k_per_round: int = 1
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Bucket:
    """Group raw terms into a smaller set of mutually exclusive buckets."""

    def __init__(self, cfg: BucketConfig, template: Optional[PromptTemplate] = None) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("bucket_prompt.jinja2")

    async def _parse(self, raw: Any) -> Dict[str, str]:
        obj = await safest_json(raw)
        if isinstance(obj, list) and obj:
            obj = obj[0]
        if isinstance(obj, dict):
            return {str(k): str(v) if v is not None else "" for k, v in obj.items()}
        return {}

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        terms_raw = [str(t) for t in df_proc[column_name].dropna().tolist()]

        # Deduplicate while preserving order
        seen: set[str] = set()
        terms: List[str] = []
        for t in terms_raw:
            if t not in seen:
                seen.add(t)
                terms.append(t)

        if not terms:
            return pd.DataFrame(columns=["bucket", "definition"])

        # ── 1: generate bucket candidates ───────────────────────────────
        prompts: List[str] = []
        ids: List[str] = []
        for rep in range(self.cfg.repeat_bucketing):
            random.shuffle(terms)
            chunks = [
                terms[i : i + self.cfg.n_terms_per_prompt]
                for i in range(0, len(terms), self.cfg.n_terms_per_prompt)
            ]
            for ci, chunk in enumerate(chunks):
                prompts.append(
                    self.template.render(
                        terms=chunk,
                        bucket_count=self.cfg.bucket_count,
                        differentiate=self.cfg.differentiate,
                        additional_instructions=self.cfg.additional_instructions or "",
                        voting=False,
                    )
                )
                ids.append(f"gen|{rep}|{ci}")

        gen_df = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=self.cfg.n_parallels,
            model=self.cfg.model,
            save_path=os.path.join(self.cfg.save_dir, "bucket_generation.csv"),
            use_dummy=self.cfg.use_dummy,
            max_timeout=self.cfg.max_timeout,
            json_mode=True,
            reset_files=reset_files,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            **kwargs,
        )
        if not isinstance(gen_df, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")

        resp_map = dict(zip(gen_df.Identifier, gen_df.Response))
        parsed = await asyncio.gather(*[self._parse(resp_map.get(i, "")) for i in ids])
        candidate_defs: Dict[str, str] = {}
        for res in parsed:
            for b, j in res.items():
                candidate_defs.setdefault(b, j)

        candidates = list(candidate_defs.keys())

        # helper to build voting prompts
        def _vote_prompts(opts: List[str], selected: List[str], tag: str):
            pr: List[str] = []
            idn: List[str] = []
            sel_map = {b: candidate_defs.get(b, "") for b in selected}
            for rep in range(self.cfg.repeat_voting):
                random.shuffle(opts)
                chunks = [
                    opts[i : i + self.cfg.n_terms_per_prompt]
                    for i in range(0, len(opts), self.cfg.n_terms_per_prompt)
                ]
                for ci, ch in enumerate(chunks):
                    sample_terms = random.sample(
                        terms, min(len(terms), self.cfg.n_terms_per_prompt)
                    )
                    pr.append(
                        self.template.render(
                            terms=sample_terms,
                            bucket_count=self.cfg.bucket_count,
                            differentiate=self.cfg.differentiate,
                            additional_instructions=self.cfg.additional_instructions
                            or "",
                            voting=True,
                            bucket_candidates=ch,
                            selected_buckets=sel_map,
                        )
                    )
                    idn.append(f"vote|{tag}|{rep}|{ci}")
            return pr, idn

        # ── 2: iterative reduction ─────────────────────────────────────
        current = candidates[:]
        round_idx = 0
        while len(current) >= 3 * self.cfg.bucket_count:
            round_idx += 1
            pr, idn = _vote_prompts(current, [], f"reduce{round_idx}")
            vote_df = await get_all_responses(
                prompts=pr,
                identifiers=idn,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=os.path.join(
                    self.cfg.save_dir, f"vote_reduce{round_idx}.csv"
                ),
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                json_mode=True,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            vote_map = dict(zip(vote_df.Identifier, vote_df.Response))
            parsed_votes = await asyncio.gather(
                *[self._parse(vote_map.get(i, "")) for i in idn]
            )
            tallies: Dict[str, int] = defaultdict(int)
            for res in parsed_votes:
                for b in res.keys():
                    tallies[b] += 1
            current.sort(
                key=lambda x: (tallies.get(x, 0), random.random()), reverse=True
            )
            keep = max(
                self.cfg.bucket_count, int(len(current) * self.cfg.next_round_frac)
            )
            current = current[:keep]

        # ── 3: final selection ─────────────────────────────────────────
        selected: List[str] = []
        remaining = current[:]
        loop_idx = 0
        while len(selected) < self.cfg.bucket_count and remaining:
            loop_idx += 1
            pr, idn = _vote_prompts(
                [o for o in remaining if o not in selected],
                selected,
                f"final{loop_idx}",
            )
            vote_df = await get_all_responses(
                prompts=pr,
                identifiers=idn,
                n_parallels=self.cfg.n_parallels,
                model=self.cfg.model,
                save_path=os.path.join(
                    self.cfg.save_dir, f"vote_final{loop_idx}.csv"
                ),
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                json_mode=True,
                reset_files=reset_files,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                **kwargs,
            )
            vote_map = dict(zip(vote_df.Identifier, vote_df.Response))
            parsed_votes = await asyncio.gather(
                *[self._parse(vote_map.get(i, "")) for i in idn]
            )
            tallies: Dict[str, int] = defaultdict(int)
            for res in parsed_votes:
                for b in res.keys():
                    if b not in selected:
                        tallies[b] += 1
            remaining = [o for o in remaining if o not in selected]
            remaining.sort(
                key=lambda x: (tallies.get(x, 0), random.random()), reverse=True
            )
            n_pick = min(
                self.cfg.top_k_per_round,
                self.cfg.bucket_count - len(selected),
                len(remaining),
            )
            winners = remaining[:n_pick]
            selected.extend(winners)

        bucket_defs = {b: candidate_defs.get(b, "") for b in selected}
        out_df = pd.DataFrame(
            {"bucket": list(bucket_defs.keys()), "definition": list(bucket_defs.values())}
        )
        out_df.to_csv(
            os.path.join(self.cfg.save_dir, self.cfg.file_name), index=False
        )
        return out_df

