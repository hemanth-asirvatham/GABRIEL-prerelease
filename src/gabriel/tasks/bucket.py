from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from ..core.prompt_template import PromptTemplate, resolve_template
from ..utils.openai_utils import get_all_responses
from ..utils import safest_json
from ..utils.logging import announce_prompt_rendering


@dataclass
class BucketConfig:
    """Configuration for :class:`Bucket`."""

    bucket_count: int = 10
    save_dir: str = "buckets"
    file_name: str = "bucket_definitions.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 650
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    additional_instructions: Optional[str] = None
    differentiate: bool = False
    n_terms_per_prompt: int = 250
    repeat_bucketing: int = 5
    repeat_voting: int = 25
    next_round_frac: float = 0.25
    top_k_per_round: int = 1
    raw_term_definitions: bool = True
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None

    def __post_init__(self) -> None:
        if self.additional_instructions is not None:
            cleaned = str(self.additional_instructions).strip()
            self.additional_instructions = cleaned or None


class Bucket:
    """Group raw terms into a smaller set of mutually exclusive buckets."""

    def __init__(
        self,
        cfg: BucketConfig,
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
            reference_filename="bucket_prompt.jinja2",
        )

    # ------------------------------------------------------------------
    # Helpers for persisting intermediate progress
    # ------------------------------------------------------------------
    def _state_path(self) -> str:
        return os.path.join(self.cfg.save_dir, "bucket_state.json")

    def _read_state(self) -> Dict[str, Any]:
        path = self._state_path()
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except FileNotFoundError:
            return {}
        except Exception:
            return {}
        return {}

    def _write_state(self, state: Dict[str, Any]) -> None:
        path = self._state_path()
        payload = dict(state)
        payload["updated_at"] = datetime.utcnow().isoformat() + "Z"
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _terms_signature(self, terms: List[str], term_map: Dict[str, str]) -> str:
        if self.cfg.raw_term_definitions:
            entries = [f"{t}::{term_map.get(t, '')}" for t in sorted(terms)]
        else:
            entries = sorted(terms)
        joined = "||".join(entries)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

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
        cache_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        state_path = self._state_path()
        if reset_files and os.path.exists(state_path):
            try:
                os.remove(state_path)
            except Exception:
                pass
        if not reset_files and os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                if {"bucket", "definition"}.issubset(cached.columns):
                    cols = ["bucket", "definition"]
                    return cached[cols]
            except Exception:
                pass

        state: Dict[str, Any] = {} if reset_files else self._read_state()

        df_proc = df.reset_index(drop=True).copy()
        raw_entries = df_proc[column_name].dropna().tolist()

        seen: Set[str] = set()
        terms: List[str] = []
        term_map: Dict[str, str] = {}
        for entry in raw_entries:
            if isinstance(entry, dict):
                for k, v in entry.items():
                    key = str(k)
                    if key not in seen:
                        seen.add(key)
                        terms.append(key)
                    term_map.setdefault(key, str(v) if v is not None else "")
            elif isinstance(entry, list):
                for item in entry:
                    key = str(item)
                    if key not in seen:
                        seen.add(key)
                        terms.append(key)
                    term_map.setdefault(key, "")
            else:
                key = str(entry)
                if key not in seen:
                    seen.add(key)
                    terms.append(key)
                term_map.setdefault(key, "")

        if not terms:
            return pd.DataFrame(columns=["bucket", "definition"])

        signature = self._terms_signature(terms, term_map)
        if state.get("terms_signature") != signature:
            state = {"terms_signature": signature}
        else:
            state["terms_signature"] = signature

        def persist_state() -> None:
            self._write_state(state)

        if state.get("finalized") and state.get("final_buckets") is not None:
            records = state.get("final_buckets") or []
            final_df = pd.DataFrame(records)
            if not final_df.empty and not {"bucket", "definition"}.issubset(final_df.columns):
                final_df = final_df.rename(columns={0: "bucket", 1: "definition"})
            if not final_df.empty and not os.path.exists(cache_path):
                try:
                    final_df.to_csv(cache_path, index=False)
                except Exception:
                    pass
            if not final_df.empty:
                cols = ["bucket", "definition"]
                return final_df[cols]
            return pd.DataFrame(columns=["bucket", "definition"])

        # ── 1: generate bucket candidates ───────────────────────────────
        candidate_defs: Dict[str, str] = {}
        candidates: List[str] = []
        if state.get("candidate_defs") and isinstance(state["candidate_defs"], dict):
            cached_defs = {
                str(k): str(v) if v is not None else ""
                for k, v in state["candidate_defs"].items()
            }
            candidate_defs.update(cached_defs)
            candidates = [
                c
                for c in state.get("candidates", list(candidate_defs.keys()))
                if c in candidate_defs
            ]
        if not candidate_defs:
            prompts: List[str] = []
            ids: List[str] = []
            chunks_per_rep = max(
                1, math.ceil(len(terms) / self.cfg.n_terms_per_prompt)
            )
            announce_prompt_rendering(
                "Bucket:generate",
                chunks_per_rep * self.cfg.repeat_bucketing,
            )
            for rep in range(self.cfg.repeat_bucketing):
                random.shuffle(terms)
                chunks = [
                    terms[i : i + self.cfg.n_terms_per_prompt]
                    for i in range(0, len(terms), self.cfg.n_terms_per_prompt)
                ]
                for ci, chunk in enumerate(chunks):
                    chunk_data = (
                        {t: term_map.get(t, "") for t in chunk}
                        if self.cfg.raw_term_definitions
                        else chunk
                    )
                    prompts.append(
                        self.template.render(
                            terms=chunk_data,
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
            for res in parsed:
                for b, j in res.items():
                    candidate_defs.setdefault(b, j)

            candidates = list(candidate_defs.keys())
            state["candidate_defs"] = candidate_defs
            state["candidates"] = candidates
            state["stage"] = "candidates"
            persist_state()
        elif not candidates:
            candidates = list(candidate_defs.keys())

        # helper to build voting prompts
        def _vote_prompts(opts: List[str], selected: List[str], tag: str):
            pr: List[str] = []
            idn: List[str] = []
            chunks_per_rep = max(
                1, math.ceil(len(opts) / self.cfg.n_terms_per_prompt)
            )
            announce_prompt_rendering(
                f"Bucket:{tag}",
                chunks_per_rep * self.cfg.repeat_voting,
            )
            for rep in range(self.cfg.repeat_voting):
                random.shuffle(opts)
                chunks = [
                    opts[i : i + self.cfg.n_terms_per_prompt]
                    for i in range(0, len(opts), self.cfg.n_terms_per_prompt)
                ]
                for ci, ch in enumerate(chunks):
                    sample_list = random.sample(
                        terms, min(len(terms), self.cfg.n_terms_per_prompt)
                    )
                    sample_terms = (
                        {t: term_map.get(t, "") for t in sample_list}
                        if self.cfg.raw_term_definitions
                        else sample_list
                    )
                    selected_map = {
                        b: candidate_defs.get(b, "") for b in selected
                    }
                    pr.append(
                        self.template.render(
                            terms=sample_terms,
                            bucket_count=self.cfg.bucket_count,
                            differentiate=self.cfg.differentiate,
                            additional_instructions=self.cfg.additional_instructions
                            or "",
                            voting=True,
                            bucket_candidates=ch,
                            selected_buckets=selected_map if selected_map else None,
                        )
                    )
                    idn.append(f"vote|{tag}|{rep}|{ci}")
            return pr, idn

        # ── 2: iterative reduction ─────────────────────────────────────
        current = candidates[:]
        if state.get("current_candidates"):
            saved_current = [
                c for c in state["current_candidates"] if c in candidate_defs
            ]
            if saved_current:
                current = saved_current
        round_idx = int(state.get("reduce_round", 0))
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
            state["current_candidates"] = current
            state["reduce_round"] = round_idx
            state["stage"] = "reduce"
            persist_state()

        # ── 3: final selection ─────────────────────────────────────────
        selected: List[str] = [
            c for c in state.get("selected", []) if c in candidate_defs
        ]
        remaining = [o for o in current if o not in selected]
        loop_idx = int(state.get("final_loop", 0))
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
            state["selected"] = selected
            state["remaining_candidates"] = remaining
            state["final_loop"] = loop_idx
            state["stage"] = "finalizing"
            persist_state()

        bucket_defs = {b: candidate_defs.get(b, "") for b in selected}
        out_df = pd.DataFrame(
            {"bucket": list(bucket_defs.keys()), "definition": list(bucket_defs.values())}
        )
        out_df.to_csv(
            os.path.join(self.cfg.save_dir, self.cfg.file_name), index=False
        )
        state["final_buckets"] = out_df.to_dict(orient="records")
        state["finalized"] = True
        state["stage"] = "complete"
        persist_state()
        return out_df
