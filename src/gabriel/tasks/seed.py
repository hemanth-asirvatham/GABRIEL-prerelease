from __future__ import annotations

import asyncio
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from gabriel.core.prompt_template import PromptTemplate, resolve_template
from gabriel.utils import safest_json
from gabriel.utils.openai_utils import get_all_responses


@dataclass
class SeedConfig:
    """Configuration options for :class:`Seed`."""

    instructions: str
    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "seed_entities.csv"
    model: str = "o3-mini"
    n_parallels: int = 400
    num_entities: int = 1000
    entities_per_generation: int = 50
    entity_batch_frac: float = 0.2
    existing_entities_cap: int = 100
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Seed:
    """Generate structured entity seeds via batched language-model calls."""

    def __init__(
        self,
        cfg: SeedConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        if cfg.num_entities <= 0:
            raise ValueError("num_entities must be positive")
        if cfg.entities_per_generation <= 0:
            raise ValueError("entities_per_generation must be positive")
        if not 0 < cfg.entity_batch_frac <= 1:
            raise ValueError("entity_batch_frac must be between 0 and 1")
        if cfg.existing_entities_cap < 0:
            raise ValueError("existing_entities_cap must be non-negative")
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="seed.jinja2",
        )

    async def run(
        self,
        *,
        existing_entities: Optional[Sequence[str]] = None,
        reset_files: bool = False,
        **response_kwargs: Any,
    ) -> pd.DataFrame:
        """Generate ``num_entities`` unique seed entities."""

        normalized_existing = self._prepare_initial_existing(existing_entities)
        seen: dict[str, str] = {}
        for ent in normalized_existing:
            norm = self._normalize_entity(ent)
            if norm and norm not in seen:
                seen[norm] = ent

        batch_target = max(
            self.cfg.entities_per_generation,
            math.ceil(self.cfg.num_entities * self.cfg.entity_batch_frac),
        )
        raw_save = os.path.join(self.cfg.save_dir, "seed_raw_responses.csv")
        batch_index = 0
        reset_next = reset_files
        while len(seen) < self.cfg.num_entities:
            remaining = self.cfg.num_entities - len(seen)
            current_goal = min(batch_target, remaining)
            current_goal = max(current_goal, self.cfg.entities_per_generation)
            prompts, identifiers = self._build_prompts(
                current_goal,
                batch_index,
                list(seen.values()),
            )
            if not prompts:
                break
            print(
                f"[Seed] Requesting {len(prompts)} prompts (batch {batch_index}, "
                f"targeting {current_goal} entities)."
            )
            kwargs = dict(response_kwargs)
            kwargs.setdefault("model", self.cfg.model)
            kwargs.setdefault("n_parallels", self.cfg.n_parallels)
            kwargs.setdefault("use_dummy", self.cfg.use_dummy)
            kwargs.setdefault("max_timeout", self.cfg.max_timeout)
            kwargs.setdefault("reasoning_effort", self.cfg.reasoning_effort)
            kwargs.setdefault("reasoning_summary", self.cfg.reasoning_summary)
            kwargs.setdefault("json_mode", True)
            kwargs.setdefault("save_path", raw_save)
            kwargs.setdefault("reset_files", reset_next)
            df_resp = await get_all_responses(
                prompts=prompts,
                identifiers=identifiers,
                **kwargs,
            )
            if not isinstance(df_resp, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")
            resp_lookup = dict(zip(df_resp.Identifier, df_resp.Response))
            parsed = await asyncio.gather(
                *[
                    self._parse_entities(resp_lookup.get(identifier, ""))
                    for identifier in identifiers
                ]
            )
            added = 0
            for ident, entity_list in zip(identifiers, parsed):
                for position, entity in enumerate(entity_list):
                    norm = self._normalize_entity(entity)
                    if not norm or norm in seen:
                        continue
                    seen[norm] = entity
                    added += 1
            print(
                f"[Seed] Added {added} new entities in batch {batch_index}. Total so far: {len(seen)}."
            )
            batch_index += 1
            reset_next = False
            if added == 0 and not any(parsed):
                break

        ordered = [seen[norm] for norm in seen]
        trimmed = ordered[: self.cfg.num_entities]
        df = pd.DataFrame(
            {
                "entity": trimmed,
                "entity_id": [f"entity-{idx:05d}" for idx in range(len(trimmed))],
            }
        )
        df["source_batch"] = df.index // max(self.cfg.entities_per_generation, 1)
        df["source_identifier"] = ["seed" for _ in range(len(trimmed))]
        final_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        df.to_csv(final_path, index=False)
        print(
            f"[Seed] Generated {len(df)} entities. Saved aggregated seeds to {final_path}."
        )
        return df

    async def _parse_entities(self, raw: Any) -> List[str]:
        obj = await safest_json(raw)
        results: List[str] = []
        if isinstance(obj, dict):
            for key in sorted(obj.keys()):
                value = obj.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    results.append(text)
        elif isinstance(obj, list):
            for item in obj:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    results.append(text)
        elif isinstance(obj, str):
            text = obj.strip()
            if text:
                results.append(text)
        return results

    def _prepare_initial_existing(
        self, entries: Optional[Sequence[str]]
    ) -> List[str]:
        if not entries:
            return []
        unique: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            text = str(entry).strip()
            if not text:
                continue
            norm = self._normalize_entity(text)
            if norm and norm not in seen:
                seen.add(norm)
                unique.append(text)
        return unique

    def _build_prompts(
        self,
        goal: int,
        batch_index: int,
        seen_entities: Sequence[str],
    ) -> Tuple[List[str], List[str]]:
        prompts: List[str] = []
        identifiers: List[str] = []
        per_call = self.cfg.entities_per_generation
        prompt_count = math.ceil(goal / per_call)
        existing_sample = self._sample_existing(seen_entities)
        existing_blob = "\n".join(existing_sample) if existing_sample else None
        for call_index in range(prompt_count):
            identifiers.append(f"seed|{batch_index}|{call_index}")
            prompts.append(
                self.template.render(
                    instructions=self.cfg.instructions,
                    entities_per_generation=per_call,
                    existing_entities=existing_blob,
                )
            )
        return prompts, identifiers

    def _sample_existing(self, seen_entities: Sequence[str]) -> List[str]:
        if not seen_entities:
            return []
        cap = max(0, self.cfg.existing_entities_cap)
        if cap == 0:
            return []
        if len(seen_entities) <= cap:
            return list(seen_entities)
        return random.sample(list(seen_entities), cap)

    @staticmethod
    def _normalize_entity(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.strip()).lower()
        return collapsed
