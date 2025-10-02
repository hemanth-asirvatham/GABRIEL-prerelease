from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils.openai_utils import get_all_responses
from gabriel.tasks.rank import Rank, RankConfig
from gabriel.tasks.rate import Rate, RateConfig


_DEF_ATTR_LABEL = "novel and brilliant"
_DEF_ATTR_DESCRIPTION = (
    "Measures how original, well-reasoned, and consequential the proposed theory is. "
    "High scores correspond to ideas that introduce a specific new mechanism, twist, addendum, etc, "
    "inspired by existing literature but genuinely novel and creative thought, "
    "and above all is just a genuinely superior scientific theory pursuant to the topic. "
    "More novel and brilliant theories say something new and interesting, capturing something in the real world better "
    "than existing thought. Ideally novel yet specific, testable, realistic, and brilliant/inspired such that top professors would admire it deeply; "
    "a high standard requiring deep thought and consideration, worthy of evaluating frontier research theories."
)


def _default_attributes() -> Dict[str, str]:
    return {_DEF_ATTR_LABEL: _DEF_ATTR_DESCRIPTION}


@dataclass
class IdeateConfig:
    """Configuration for :class:`Ideate`."""

    save_dir: str = os.path.expanduser("~/Documents/runs")
    file_name: str = "ideation.csv"
    model: str = "gpt-5-mini"
    ranking_model: Optional[str] = None
    n_parallels: int = 750
    n_ideas: int = 1000
    evaluation_mode: str = "recursive_rank"
    attributes: Dict[str, str] = field(default_factory=_default_attributes)
    rank_attribute: Optional[str] = None
    recursive_fraction: float = 1.0 / 3.0
    recursive_min_remaining: int = 30
    recursive_final_round_multiplier: int = 3
    recursive_cut_side: str = "top"
    recursive_rate_first_round: bool = True
    additional_instructions: Optional[str] = None
    use_dummy: bool = False
    web_search: bool = False
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Ideate:
    """Generate and optionally score frontier scientific theories."""

    def __init__(
        self,
        cfg: IdeateConfig,
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
                template_path, reference_filename="ideation_prompt.jinja2"
            )
        self.template = template or PromptTemplate.from_package("ideation_prompt.jinja2")

    async def run(
        self,
        topic: str,
        *,
        additional_instructions: Optional[str] = None,
        evaluation_mode: Optional[str] = None,
        attributes: Optional[Dict[str, str]] = None,
        rank_attribute: Optional[str] = None,
        reset_files: bool = False,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        rank_config_updates: Optional[Dict[str, Any]] = None,
        rank_run_kwargs: Optional[Dict[str, Any]] = None,
        rate_config_updates: Optional[Dict[str, Any]] = None,
        rate_run_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate a large batch of theories and optionally score them."""

        base_name = os.path.splitext(self.cfg.file_name)[0]
        final_path = os.path.join(self.cfg.save_dir, f"{base_name}_final.csv")

        if not reset_files and os.path.exists(final_path):
            try:
                print(f"[Ideate] Loading cached results from {final_path}")
                cached = pd.read_csv(final_path)
                return cached
            except Exception:
                print("[Ideate] Failed to load cached results; recomputing.")

        attrs = attributes or self.cfg.attributes
        if not attrs:
            raise ValueError("At least one attribute must be provided for scoring")
        attr_key = rank_attribute or self.cfg.rank_attribute or next(iter(attrs))

        mode = (evaluation_mode or self.cfg.evaluation_mode or "none").lower()
        if mode not in {"recursive_rank", "rank", "rate", "none"}:
            raise ValueError(
                "evaluation_mode must be one of 'recursive_rank', 'rank', 'rate', or 'none'"
            )

        gen_kwargs = dict(generation_kwargs or {})
        rank_cfg_updates = dict(rank_config_updates or {})
        rank_run_kwargs = dict(rank_run_kwargs or {})
        rate_cfg_updates = dict(rate_config_updates or {})
        rate_run_kwargs = dict(rate_run_kwargs or {})

        raw_df = await self._generate_reports(
            topic,
            additional_instructions or self.cfg.additional_instructions,
            reset_files=reset_files,
            **gen_kwargs,
        )
        parsed_df = self._parse_reports(raw_df, topic)

        if mode == "none":
            parsed_df.to_csv(final_path, index=False)
            return parsed_df

        if mode == "rate":
            scored_df = await self._apply_rate(
                parsed_df,
                attrs,
                attr_key,
                reset_files=reset_files,
                config_updates=rate_cfg_updates,
                run_kwargs=rate_run_kwargs,
            )
        else:
            recursive = mode == "recursive_rank"
            scored_df = await self._apply_rank(
                parsed_df,
                attrs,
                attr_key,
                recursive=recursive,
                reset_files=reset_files,
                config_updates=rank_cfg_updates,
                run_kwargs=rank_run_kwargs,
            )

        scored_df.to_csv(final_path, index=False)
        return scored_df

    async def _generate_reports(
        self,
        topic: str,
        additional_instructions: Optional[str],
        *,
        reset_files: bool,
        **generation_kwargs: Any,
    ) -> pd.DataFrame:
        base_name = os.path.splitext(self.cfg.file_name)[0]
        raw_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")
        print(
            f"[Ideate] Generating {self.cfg.n_ideas} theories with model {self.cfg.model}."
        )

        prompts = [
            self.template.render(
                topic=topic,
                additional_instructions=additional_instructions or "",
            )
            for _ in range(self.cfg.n_ideas)
        ]
        identifiers = [f"idea-{i:05d}" for i in range(len(prompts))]

        kwargs = dict(
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            save_path=raw_path,
            reset_files=reset_files,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            print_example_prompt=True,
        )
        kwargs.update(generation_kwargs)
        if "web_search" not in kwargs:
            kwargs["web_search"] = self.cfg.web_search

        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            **kwargs,
        )
        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")
        df_resp = df_resp.copy()
        df_resp["idea_id"] = df_resp["Identifier"].astype(str)
        df_resp["topic"] = topic
        df_resp["report_text"] = df_resp["Response"].astype(str).str.strip()
        return df_resp

    def _parse_reports(self, df: pd.DataFrame, topic: str) -> pd.DataFrame:
        print("[Ideate] Parsing structured sections from each report.")
        df_proc = df.copy()
        df_proc["report_text"] = df_proc["report_text"].astype(str).str.strip()

        sections: Dict[str, List[Optional[str]]] = {
            "title": [],
            "in_a_nutshell": [],
            "in_one_paragraph": [],
            "full_thinking": [],
            "summary_preview": [],
        }

        for text in df_proc["report_text"].astype(str):
            parsed = self._extract_sections(text)
            sections["title"].append(parsed.get("title"))
            sections["in_a_nutshell"].append(parsed.get("in_a_nutshell"))
            sections["in_one_paragraph"].append(parsed.get("in_one_paragraph"))
            sections["full_thinking"].append(parsed.get("full_thinking"))
            preview_parts: List[str] = []
            for key, label in [
                ("title", "Title"),
                ("in_a_nutshell", "In a nutshell"),
                ("in_one_paragraph", "In one paragraph"),
            ]:
                value = parsed.get(key)
                if value:
                    preview_parts.append(f"{label}: {value}")
            sections["summary_preview"].append("\n\n".join(preview_parts) if preview_parts else None)

        for key, values in sections.items():
            df_proc[key] = values

        df_proc["topic"] = topic
        return df_proc

    def _extract_sections(self, text: str) -> Dict[str, Optional[str]]:
        headers = {
            "title": "title",
            "in a nutshell": "in_a_nutshell",
            "in one paragraph": "in_one_paragraph",
            "the full thinking": "full_thinking",
        }
        result: Dict[str, Optional[str]] = {v: None for v in headers.values()}
        current_key: Optional[str] = None
        buffer: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line and current_key is None:
                continue
            lowered = line.lower()
            matched = None
            for header_text, key in headers.items():
                if lowered.startswith(f"{header_text}:"):
                    matched = key
                    content = line[len(header_text) + 1 :].strip()
                    if current_key is not None:
                        result[current_key] = "\n".join(buffer).strip() or None
                    buffer = [content] if content else []
                    current_key = key
                    break
            if matched is None:
                if current_key is not None:
                    buffer.append(raw_line.rstrip())
        if current_key is not None:
            result[current_key] = "\n".join(buffer).strip() or None
        return result

    async def _apply_rate(
        self,
        df: pd.DataFrame,
        attributes: Dict[str, str],
        attr_key: str,
        *,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        print("[Ideate] Scoring reports with Rate task.")
        rate_save = os.path.join(self.cfg.save_dir, "rate")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        cfg_kwargs: Dict[str, Any] = dict(
            attributes=attributes,
            save_dir=rate_save,
            file_name=f"{base_name}_ratings.csv",
            model=self.cfg.ranking_model or self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        cfg_kwargs.update(config_updates)
        rate_cfg = RateConfig(**cfg_kwargs)
        rate_task = Rate(rate_cfg)
        rate_run_opts = dict(run_kwargs)
        rate_run_opts.setdefault("web_search", False)
        df_scored = await rate_task.run(
            df,
            "report_text",
            reset_files=reset_files,
            **rate_run_opts,
        )
        return self._sort_results(df_scored, attr_key)

    async def _apply_rank(
        self,
        df: pd.DataFrame,
        attributes: Dict[str, str],
        attr_key: str,
        *,
        recursive: bool,
        reset_files: bool,
        config_updates: Dict[str, Any],
        run_kwargs: Dict[str, Any],
    ) -> pd.DataFrame:
        print("[Ideate] Ranking reports with Rank task.")
        rank_save = os.path.join(self.cfg.save_dir, "rank")
        base_name = os.path.splitext(self.cfg.file_name)[0]
        cfg_kwargs: Dict[str, Any] = dict(
            attributes=attributes,
            save_dir=rank_save,
            file_name=f"{base_name}_rankings",
            model=self.cfg.ranking_model or self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            recursive=recursive,
            recursive_fraction=self.cfg.recursive_fraction,
            recursive_min_remaining=self.cfg.recursive_min_remaining,
            recursive_final_round_multiplier=self.cfg.recursive_final_round_multiplier,
            recursive_cut_side=self.cfg.recursive_cut_side,
            recursive_rate_first_round=self.cfg.recursive_rate_first_round,
        )
        if attr_key and cfg_kwargs.get("recursive"):
            cfg_kwargs.setdefault("recursive_cut_attr", attr_key)
        cfg_kwargs.update(config_updates)
        rank_cfg = RankConfig(**cfg_kwargs)
        rank_task = Rank(rank_cfg)
        rank_run_opts = dict(run_kwargs)
        rank_run_opts.setdefault("web_search", False)
        df_ranked = await rank_task.run(
            df,
            "report_text",
            id_column="idea_id",
            reset_files=reset_files,
            **rank_run_opts,
        )
        return self._sort_results(df_ranked, attr_key)

    def _sort_results(self, df: pd.DataFrame, attr_key: str) -> pd.DataFrame:
        if attr_key not in df.columns:
            return df.reset_index(drop=True)
        df_sorted = df.sort_values(by=attr_key, ascending=False, na_position="last").copy()
        df_sorted.reset_index(drop=True, inplace=True)
        rank_col = f"{attr_key}_rank"
        positions: List[Optional[int]] = []
        counter = 1
        for value in df_sorted[attr_key]:
            if pd.isna(value):
                positions.append(None)
            else:
                positions.append(counter)
                counter += 1
        df_sorted[rank_col] = pd.Series(positions, dtype="Int64")
        return df_sorted
