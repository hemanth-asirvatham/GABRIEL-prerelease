from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses


@dataclass
class ParaphraseConfig:
    """Configuration for :class:`Paraphrase`."""

    instructions: str
    revised_column_name: Optional[str] = None
    n_revisions: int = 1
    save_dir: str = "paraphrase"
    file_name: str = "paraphrase_responses.csv"
    model: str = "gpt-5-mini"
    json_mode: bool = False
    use_web_search: bool = False
    n_parallels: int = 400
    use_dummy: bool = False
    reasoning_effort: Optional[str] = None
    include_summaries: bool = False


class Paraphrase:
    """Paraphrase text columns in a DataFrame."""

    def __init__(self, cfg: ParaphraseConfig, template: Optional[PromptTemplate] = None) -> None:
        self.cfg = cfg
        self.template = template or PromptTemplate.from_package("paraphrase_prompt.jinja2")
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Paraphrase ``df[column_name]`` and return a DataFrame with revisions."""

        df_proc = df.reset_index(drop=True).copy()
        texts = df_proc[column_name].astype(str).tolist()
        base_col = self.cfg.revised_column_name or f"{column_name}_revised"
        n = self.cfg.n_revisions if self.cfg.n_revisions and self.cfg.n_revisions > 0 else 1

        prompts: list[str] = []
        identifiers: list[str] = []
        for idx, text in enumerate(texts):
            for j in range(1, n + 1):
                prompts.append(
                    self.template.render(text=text, instructions=self.cfg.instructions)
                )
                identifiers.append(f"row_{idx}_rev{j}")

        save_path = os.path.join(self.cfg.save_dir, self.cfg.file_name)
        resp_df = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            save_path=save_path,
            model=self.cfg.model,
            json_mode=self.cfg.json_mode,
            use_web_search=self.cfg.use_web_search,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reset_files=reset_files,
            reasoning_effort=self.cfg.reasoning_effort,
            include_summaries=self.cfg.include_summaries,
            **kwargs,
        )

        resp_map: Dict[tuple[int, int], str] = {}
        for ident, resp in zip(resp_df["Identifier"], resp_df["Response"]):
            main = resp[0] if isinstance(resp, list) and resp else resp
            m = re.match(r"row_(\d+)_rev(\d+)", ident)
            if m:
                row = int(m.group(1))
                rev = int(m.group(2)) - 1
                resp_map[(row, rev)] = main

        col_names = [base_col] if n == 1 else [f"{base_col}_{i}" for i in range(1, n + 1)]
        for j, col in enumerate(col_names):
            df_proc[col] = [resp_map.get((i, j), "") for i in range(len(df_proc))]

        out_path = os.path.join(
            self.cfg.save_dir,
            f"{os.path.splitext(self.cfg.file_name)[0]}_cleaned.csv",
        )
        df_proc.to_csv(out_path, index=False)
        return df_proc

