"""Lightweight runner for arbitrary prompts via :func:`get_all_responses`."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd

from ..utils import load_audio_inputs, load_image_inputs
from ..utils.openai_utils import get_all_responses, response_to_text
from ..utils.parsing import safe_json


@dataclass
class WhateverConfig:
    """Configuration for :class:`Whatever`."""

    save_dir: str = "whatever"
    file_name: str = "custom_prompt_responses.csv"
    model: str = "gpt-5-mini"
    json_mode: bool = False
    web_search: Optional[bool] = None
    web_search_filters: Optional[Dict[str, Any]] = None
    search_context_size: str = "medium"
    n_parallels: int = 750
    use_dummy: bool = False
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None


class Whatever:
    """Prepare prompts and dispatch them through :func:`get_all_responses`."""

    def __init__(self, cfg: WhateverConfig) -> None:
        expanded = os.path.expandvars(os.path.expanduser(cfg.save_dir))
        os.makedirs(expanded, exist_ok=True)
        cfg.save_dir = expanded
        self.cfg = cfg

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_identifiers(
        prompts: List[str], provided: Optional[List[str]] = None
    ) -> List[str]:
        if provided is not None:
            if len(provided) != len(prompts):
                raise ValueError("Length of identifiers must match number of prompts")
            ids = [str(i) for i in provided]
            if len(set(ids)) != len(ids):
                raise ValueError("Identifiers must be unique")
            return ids

        counts: Dict[str, int] = {}
        generated: List[str] = []
        for prompt in prompts:
            key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
            idx = counts.get(key, 0)
            counts[key] = idx + 1
            ident = key if idx == 0 else f"{key}-{idx}"
            generated.append(ident)
        return generated

    # ------------------------------------------------------------------
    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        try:
            res = pd.isna(value)
        except Exception:
            return False
        try:
            return bool(res)
        except Exception:
            return False

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_domains(value: Any) -> List[str]:
        if Whatever._is_missing(value):
            return []
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, dict):
            return []
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            domains = [str(item).strip() for item in value if str(item).strip()]
            return domains
        text = str(value).strip()
        return [text] if text else []

    # ------------------------------------------------------------------
    @staticmethod
    def extract_json(
        df: pd.DataFrame,
        *,
        text_column: str = "Response",
        target_column: str = "Response JSON",
    ) -> pd.DataFrame:
        """Return a copy of ``df`` with parsed JSON objects in ``target_column``."""

        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found in DataFrame; available columns: {list(df.columns)}"
            )

        out = df.copy()

        def _parse(value: Any) -> Optional[Union[dict, list]]:
            text = response_to_text(value)
            if not text:
                return None
            parsed = safe_json(text)
            return parsed if parsed else None

        out[target_column] = out[text_column].apply(_parse)
        return out

    async def run(
        self,
        data: Union[str, List[str], pd.DataFrame],
        *,
        identifiers: Optional[List[str]] = None,
        column_name: Optional[str] = None,
        identifier_column: Optional[str] = None,
        image_column: Optional[str] = None,
        audio_column: Optional[str] = None,
        prompt_images: Optional[Dict[str, List[str]]] = None,
        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
        web_search_filters: Optional[Dict[str, Any]] = None,
        reset_files: bool = False,
        parse_json: Optional[bool] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Normalise inputs and call :func:`get_all_responses`.

        Parameters
        ----------
        parse_json:
            When ``True`` (the default while ``json_mode`` is enabled) the
            returned DataFrame will include a ``Response JSON`` column with the
            parsed structure for each response.  Set to ``False`` to skip the
            extra parsing step and keep the raw JSON text in the ``Response``
            column.
        """

        filters_spec: Dict[str, Any] = dict(
            web_search_filters
            if web_search_filters is not None
            else (self.cfg.web_search_filters or {})
        )

        df_filters: Optional[Dict[str, Dict[str, Any]]] = None
        global_filters: Optional[Dict[str, Any]] = filters_spec or None

        if isinstance(data, pd.DataFrame):
            if column_name is None:
                raise ValueError("column_name must be provided when passing a DataFrame")
            if column_name not in data.columns:
                raise ValueError(f"Column '{column_name}' not found in DataFrame")
            df = data.reset_index(drop=True)
            prompt_series = df[column_name]
            prompt_values = [
                "" if self._is_missing(val) else str(val)
                for val in prompt_series.tolist()
            ]
            if identifier_column is not None:
                if identifier_column not in df.columns:
                    raise ValueError(
                        f"Identifier column '{identifier_column}' not found in DataFrame"
                    )
                identifiers_list = [str(i) for i in df[identifier_column].tolist()]
                if len(set(identifiers_list)) != len(identifiers_list):
                    raise ValueError("identifier_column must contain unique values")
            else:
                identifiers_list = self._generate_identifiers(prompt_values)

            image_map: Dict[str, List[str]] = {}
            if image_column is not None:
                if image_column not in df.columns:
                    raise ValueError(
                        f"Image column '{image_column}' not found in DataFrame"
                    )
                for ident, cell in zip(identifiers_list, df[image_column]):
                    imgs = load_image_inputs(cell)
                    if imgs:
                        image_map[str(ident)] = imgs

            audio_map: Dict[str, List[Dict[str, str]]] = {}
            if audio_column is not None:
                if audio_column not in df.columns:
                    raise ValueError(
                        f"Audio column '{audio_column}' not found in DataFrame"
                    )
                for ident, cell in zip(identifiers_list, df[audio_column]):
                    auds = load_audio_inputs(cell)
                    if auds:
                        audio_map[str(ident)] = auds

            column_filters: Dict[str, str] = {}
            base_filters: Dict[str, Any] = {}
            for key, spec in filters_spec.items():
                if isinstance(spec, str) and spec in df.columns:
                    column_filters[key] = spec
                elif key == "allowed_domains" and isinstance(spec, str) and spec in df.columns:
                    column_filters[key] = spec
                else:
                    base_filters[key] = spec

            per_prompt_filters: Dict[str, Dict[str, Any]] = {}
            if column_filters:
                for idx, ident in enumerate(identifiers_list):
                    row = df.iloc[idx]
                    row_filters: Dict[str, Any] = {}
                    for key, col in column_filters.items():
                        value = row.get(col)
                        if self._is_missing(value):
                            continue
                        if key == "allowed_domains":
                            domains = self._coerce_domains(value)
                            if domains:
                                row_filters[key] = domains
                        else:
                            text = str(value).strip()
                            if text:
                                row_filters[key] = text
                    if row_filters:
                        per_prompt_filters[str(ident)] = row_filters
            df_filters = per_prompt_filters or None
            global_filters = base_filters or None

            prompts_list = prompt_values
        else:
            if isinstance(data, str):
                prompts_list = [data]
            else:
                prompts_list = [str(p) for p in data]
            identifiers_list = self._generate_identifiers(
                prompts_list, identifiers
            )
            image_map = {}
            audio_map = {}

        if prompt_images:
            if not isinstance(prompt_images, dict):
                raise TypeError("prompt_images must be a mapping of identifier to images")
            for key, val in prompt_images.items():
                if val:
                    image_map[str(key)] = val

        if prompt_audio:
            if not isinstance(prompt_audio, dict):
                raise TypeError("prompt_audio must be a mapping of identifier to audio payloads")
            for key, val in prompt_audio.items():
                if val:
                    audio_map[str(key)] = val

        images_payload = image_map or None
        audio_payload = audio_map or None

        save_path = kwargs.pop(
            "save_path", os.path.join(self.cfg.save_dir, self.cfg.file_name)
        )

        web_search_flag = (
            self.cfg.web_search
            if self.cfg.web_search is not None
            else bool(global_filters or df_filters)
        )

        df_resp = await get_all_responses(
            prompts=prompts_list,
            identifiers=identifiers_list,
            prompt_images=images_payload,
            prompt_audio=audio_payload,
            prompt_web_search_filters=df_filters,
            save_path=save_path,
            model=self.cfg.model,
            json_mode=self.cfg.json_mode,
            web_search=web_search_flag,
            web_search_filters=global_filters,
            search_context_size=self.cfg.search_context_size,
            n_parallels=self.cfg.n_parallels,
            use_dummy=self.cfg.use_dummy,
            reset_files=reset_files,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            **kwargs,
        )
        if not isinstance(df_resp, pd.DataFrame):
            raise RuntimeError("get_all_responses returned no DataFrame")

        df_clean = df_resp.copy()
        if "Response" in df_clean.columns:
            df_clean["Response"] = df_clean["Response"].apply(response_to_text)

        if self.cfg.json_mode:
            auto_parse = parse_json if parse_json is not None else True
            if auto_parse:
                df_clean = self.extract_json(
                    df_clean,
                    text_column="Response",
                    target_column="Response JSON",
                )
                print(
                    "[Whatever] Parsed JSON output is available in the 'Response JSON' column."
                )
            else:
                print(
                    "[Whatever] JSON responses are stored as text in the 'Response' column. "
                    "Call `Whatever.extract_json(df)` to parse them into structured objects."
                )

        return df_clean
