from __future__ import annotations
import hashlib
import os
from pathlib import Path
import re
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional
import json

import pandas as pd

from ..core.prompt_template import PromptTemplate
from ..utils.openai_utils import get_all_responses
from ..utils import (
    safest_json,
    load_image_inputs,
    load_audio_inputs,
)


def _collect_predictions(row: pd.Series) -> List[str]:
    """Return labels whose values evaluate to ``True``.

    Parameters
    ----------
    row:
        A series containing only label columns.

    Returns
    -------
    list of str
        Labels for which the value is truthy.
    """

    return [lab for lab, val in row.items() if bool(val)]


# ────────────────────────────
# Configuration dataclass
# ────────────────────────────
@dataclass
class ClassifyConfig:
    """Configuration for :class:`Classify`."""

    labels: Dict[str, str]  # {"label_name": "description", ...}
    save_dir: str = "classifier"
    file_name: str = "classify_responses.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    n_runs: int = 1
    min_frequency: float = 0.6
    additional_instructions: str = ""
    additional_guidelines: str = ""
    use_dummy: bool = False
    max_timeout: Optional[float] = None
    modality: str = "text"
    n_attributes_per_run: int = 8
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    differentiate: bool = False
    circle_first: Optional[bool] = None


# ────────────────────────────
# Main Basic classifier task
# ────────────────────────────
class Classify:
    """Robust passage classifier using an LLM.

    * Accepts a list of *texts* (not a DataFrame) just like :class:`Rate`.
    * Persists/reads cached responses via the **save_path** attribute (same pattern as
      :class:`Rate`).
    """

    _FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    # -----------------------------------------------------------------
    def __init__(
        self,
        cfg: ClassifyConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:  # noqa: D401,E501
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        if template is not None and template_path is not None:
            raise ValueError("Provide either template or template_path, not both")
        if template_path is not None:
            template = PromptTemplate.from_file(
                template_path, reference_filename="classification_prompt.jinja2"
            )
        self.template = template or PromptTemplate.from_package(
            "classification_prompt.jinja2"
        )

    # -----------------------------------------------------------------
    # Helpers for parsing raw model output
    # -----------------------------------------------------------------
    @staticmethod
    def _regex(raw: str, labels: List[str]) -> Dict[str, Optional[bool]]:
        out: Dict[str, Optional[bool]] = {}
        for lab in labels:
            pat = re.compile(
                rf'\s*"?\s*{re.escape(lab)}\s*"?\s*:\s*(true|false)', re.I | re.S
            )
            m = pat.search(raw)
            out[lab] = None if not m else m.group(1).lower() == "true"
        return out

    async def _parse(self, resp: Any, labels: List[str]) -> Dict[str, Optional[bool]]:
        # unwrap common response containers (list-of-one, bytes, fenced blocks)
        if isinstance(resp, list) and len(resp) == 1:
            resp = resp[0]
        if isinstance(resp, (bytes, bytearray)):
            resp = resp.decode()
        data: Optional[Any] = None
        if isinstance(resp, str):
            m = self._FENCE_RE.search(resp)
            if m:
                resp = m.group(1).strip()

            data = await safest_json(resp)
        elif isinstance(resp, dict):
            data = resp
        if isinstance(data, dict):
            norm = {
                k.strip().lower(): (
                    True
                    if str(v).strip().lower() in {"true", "yes", "1"}
                    else (
                        False
                        if str(v).strip().lower() in {"false", "no", "0"}
                        else None
                    )
                )
                for k, v in data.items()
            }
            return {lab: norm.get(lab.lower(), None) for lab in labels}

        # fallback to regex extraction
        return self._regex(str(resp), labels)

    # -----------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------
    async def run(
        self,
        df: pd.DataFrame,
        column_name: Optional[str] = None,
        *,
        circle_column_name: Optional[str] = None,
        square_column_name: Optional[str] = None,
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Classify items and return ``df`` with label columns."""

        if self.cfg.differentiate:
            if circle_column_name is None or square_column_name is None:
                raise ValueError(
                    "circle_column_name and square_column_name are required when differentiate is True"
                )
        elif column_name is None:
            raise ValueError("column_name is required when differentiate is False")

        df_proc = df.reset_index(drop=True).copy()

        label_items = list(self.cfg.labels.items())
        label_batches: List[Dict[str, str]] = [
            dict(label_items[i : i + self.cfg.n_attributes_per_run])
            for i in range(0, len(label_items), self.cfg.n_attributes_per_run)
        ]

        prompts: List[str] = []
        ids: List[str] = []
        base_ids: List[str] = []
        id_to_circle_first: Dict[str, bool] = {}
        id_to_rows: DefaultDict[str, List[int]] = defaultdict(list)
        id_to_val: Dict[str, Any] = {}
        prompt_texts: Dict[str, str] = {}
        prompt_circles: Dict[str, str] = {}
        prompt_squares: Dict[str, str] = {}

        if self.cfg.differentiate:
            circles = df_proc[circle_column_name].tolist()  # type: ignore[index]
            squares = df_proc[square_column_name].tolist()  # type: ignore[index]
            for row, (circ, sq) in enumerate(zip(circles, squares)):
                clean = " ".join(str(circ).split()) + "|" + " ".join(str(sq).split())
                sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]
                id_to_rows[sha8].append(row)
                if len(id_to_rows[sha8]) > 1:
                    continue
                id_to_val[sha8] = (circ, sq)
                prompt_circles[sha8] = (
                    circ if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                prompt_squares[sha8] = (
                    sq if self.cfg.modality in {"text", "entity", "web"} else ""
                )
                circle_first_flag = (
                    self.cfg.circle_first
                    if self.cfg.circle_first is not None
                    else random.random() < 0.5
                )
                id_to_circle_first[sha8] = circle_first_flag
                base_ids.append(sha8)
            for batch_idx, batch_labels in enumerate(label_batches):
                for ident in base_ids:
                    prompts.append(
                        self.template.render(
                            entry_circle=prompt_circles[ident],
                            entry_square=prompt_squares[ident],
                            attributes=batch_labels,
                            additional_instructions=self.cfg.additional_instructions,
                            additional_guidelines=self.cfg.additional_guidelines,
                            differentiate=True,
                            modality=self.cfg.modality,
                            circle_first=id_to_circle_first[ident],
                        )
                    )
                    ids.append(f"{ident}_batch{batch_idx}")
        else:
            values = df_proc[column_name].tolist()  # type: ignore[index]
            for row, val in enumerate(values):
                clean = " ".join(str(val).split())
                sha8 = hashlib.sha1(clean.encode()).hexdigest()[:8]
                id_to_rows[sha8].append(row)
                if len(id_to_rows[sha8]) > 1:
                    continue
                id_to_val[sha8] = values[row]
                prompt_texts[sha8] = (
                    str(values[row])
                    if self.cfg.modality in {"text", "entity", "web"}
                    else ""
                )
                base_ids.append(sha8)
            for batch_idx, batch_labels in enumerate(label_batches):
                for ident in base_ids:
                    prompts.append(
                        self.template.render(
                            text=prompt_texts[ident],
                            attributes=batch_labels,
                            additional_instructions=self.cfg.additional_instructions,
                            additional_guidelines=self.cfg.additional_guidelines,
                            modality=self.cfg.modality,
                        )
                    )
                    ids.append(f"{ident}_batch{batch_idx}")

        prompt_images: Optional[Dict[str, List[str]]] = None
        prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None

        if self.cfg.modality == "image":
            tmp: Dict[str, List[str]] = {}
            for ident, rows in id_to_rows.items():
                imgs: List[str] = []
                if self.cfg.differentiate:
                    circ, sq = id_to_val[ident]
                    circ_imgs = load_image_inputs(circ)
                    sq_imgs = load_image_inputs(sq)
                    if id_to_circle_first.get(ident, False):
                        if circ_imgs:
                            imgs.extend(circ_imgs)
                        if sq_imgs:
                            imgs.extend(sq_imgs)
                    else:
                        if sq_imgs:
                            imgs.extend(sq_imgs)
                        if circ_imgs:
                            imgs.extend(circ_imgs)
                else:
                    imgs = load_image_inputs(id_to_val[ident])
                if imgs:
                    for batch_idx in range(len(label_batches)):
                        tmp[f"{ident}_batch{batch_idx}"] = imgs
            prompt_images = tmp or None
        elif self.cfg.modality == "audio":
            tmp_a: Dict[str, List[Dict[str, str]]] = {}
            for ident, rows in id_to_rows.items():
                auds: List[Dict[str, str]] = []
                if self.cfg.differentiate:
                    circ, sq = id_to_val[ident]
                    circ_auds = load_audio_inputs(circ)
                    sq_auds = load_audio_inputs(sq)
                    if id_to_circle_first.get(ident, False):
                        if circ_auds:
                            auds.extend(circ_auds)
                        if sq_auds:
                            auds.extend(sq_auds)
                    else:
                        if sq_auds:
                            auds.extend(sq_auds)
                        if circ_auds:
                            auds.extend(circ_auds)
                else:
                    auds = load_audio_inputs(id_to_val[ident])
                if auds:
                    for batch_idx in range(len(label_batches)):
                        tmp_a[f"{ident}_batch{batch_idx}"] = auds
            prompt_audio = tmp_a or None

        base_name = os.path.splitext(self.cfg.file_name)[0]
        csv_path = os.path.join(self.cfg.save_dir, f"{base_name}_raw_responses.csv")
        attr_path = os.path.join(self.cfg.save_dir, f"{base_name}_attrs.json")

        if reset_files and os.path.exists(attr_path):
            try:
                os.remove(attr_path)
            except Exception:
                pass
        if os.path.exists(attr_path):
            try:
                with open(attr_path) as f:
                    saved_labels = json.load(f)
                if saved_labels != self.cfg.labels:
                    print(
                        "[Classify] Loading existing labels from save directory. If you want to use different labels, set reset_files=True or use a different save_dir."
                    )
                    print(saved_labels)
                    self.cfg.labels = saved_labels
            except Exception:
                pass
        else:
            try:
                with open(attr_path, "w") as f:
                    json.dump(self.cfg.labels, f, indent=2)
            except Exception:
                pass

        kwargs.setdefault("web_search", self.cfg.modality == "web")

        if not isinstance(self.cfg.n_runs, int) or self.cfg.n_runs < 1:
            raise ValueError("n_runs must be an integer >= 1")

        if self.cfg.n_runs == 1:
            df_resp_all = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                prompt_images=prompt_images,
                prompt_audio=prompt_audio,
                n_parallels=self.cfg.n_parallels,
                save_path=csv_path,
                reset_files=reset_files,
                json_mode=self.cfg.modality != "audio",
                model=self.cfg.model,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")
            df_resps = [df_resp_all]
        else:
            prompts_all: List[str] = []
            ids_all: List[str] = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                prompts_all.extend(prompts)
                ids_all.extend([f"{ident}_run{run_idx}" for ident in ids])

            prompt_images_all: Optional[Dict[str, List[str]]] = None
            if prompt_images:
                prompt_images_all = {}
                for ident, imgs in prompt_images.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_images_all[f"{ident}_run{run_idx}"] = imgs
            prompt_audio_all: Optional[Dict[str, List[Dict[str, str]]]] = None
            if prompt_audio:
                prompt_audio_all = {}
                for ident, auds in prompt_audio.items():
                    for run_idx in range(1, self.cfg.n_runs + 1):
                        prompt_audio_all[f"{ident}_run{run_idx}"] = auds

            df_resp_all = await get_all_responses(
                prompts=prompts_all,
                identifiers=ids_all,
                prompt_images=prompt_images_all,
                prompt_audio=prompt_audio_all,
                n_parallels=self.cfg.n_parallels,
                save_path=csv_path,
                reset_files=reset_files,
                json_mode=self.cfg.modality != "audio",
                model=self.cfg.model,
                use_dummy=self.cfg.use_dummy,
                max_timeout=self.cfg.max_timeout,
                reasoning_effort=self.cfg.reasoning_effort,
                reasoning_summary=self.cfg.reasoning_summary,
                print_example_prompt=True,
                **kwargs,
            )
            if not isinstance(df_resp_all, pd.DataFrame):
                raise RuntimeError("get_all_responses returned no DataFrame")

            df_resps = []
            for run_idx in range(1, self.cfg.n_runs + 1):
                suffix = f"_run{run_idx}"
                sub = df_resp_all[df_resp_all.Identifier.str.endswith(suffix)].copy()
                sub.Identifier = sub.Identifier.str.replace(
                    suffix + "$", "", regex=True
                )
                df_resps.append(sub)

        # parse each run and construct disaggregated records
        full_records: List[Dict[str, Any]] = []
        total_orphans = 0
        all_labels = list(self.cfg.labels.keys())
        for run_idx, df_resp in enumerate(df_resps, start=1):
            id_to_labels: Dict[str, Dict[str, Optional[bool]]] = {
                ident: {lab: None for lab in all_labels} for ident in base_ids
            }
            orphans = 0
            for ident_batch, raw in zip(df_resp.Identifier, df_resp.Response):
                if "_batch" not in ident_batch:
                    continue
                base_ident, batch_part = ident_batch.rsplit("_batch", 1)
                if base_ident not in id_to_rows:
                    orphans += 1
                    continue
                batch_idx = int(batch_part)
                labs = list(label_batches[batch_idx].keys())
                parsed = await self._parse(raw, labs)
                for lab in labs:
                    id_to_labels[base_ident][lab] = parsed.get(lab)
            total_orphans += orphans
            for ident in base_ids:
                parsed = id_to_labels.get(ident, {lab: None for lab in all_labels})
                if self.cfg.differentiate:
                    circ_val, sq_val = id_to_val[ident]
                    rec = {"circle": circ_val, "square": sq_val, "run": run_idx}
                else:
                    rec = {"text": id_to_val[ident], "run": run_idx}
                rec.update({lab: parsed.get(lab) for lab in all_labels})
                full_records.append(rec)

        if total_orphans:
            print(
                f"[Classify] WARNING: {total_orphans} response(s) had no matching passage this run."
            )

        if self.cfg.differentiate:
            full_df = pd.DataFrame(full_records).set_index(["circle", "square", "run"])
            index_cols = ["circle", "square", "run"]
            group_cols = ["circle", "square"]
        else:
            full_df = pd.DataFrame(full_records).set_index(["text", "run"])
            index_cols = ["text", "run"]
            group_cols = ["text"]
        if self.cfg.n_runs > 1:
            disagg_path = os.path.join(
                self.cfg.save_dir, f"{base_name}_full_disaggregated.csv"
            )
            full_df.to_csv(disagg_path, index_label=index_cols)

        # aggregate across runs using a minimum frequency threshold
        def _min_freq(s: pd.Series) -> Optional[bool]:
            if s.notna().sum() == 0:
                return None
            true_count = s.fillna(False).sum()
            prop = true_count / self.cfg.n_runs
            return prop >= self.cfg.min_frequency

        agg_df = pd.DataFrame(
            {
                lab: full_df[lab].groupby(group_cols).apply(_min_freq)
                for lab in self.cfg.labels
            }
        )

        filled = agg_df.dropna(how="all").shape[0]
        print(f"[Classify] Filled {filled}/{len(agg_df)} unique texts.")

        total = len(agg_df)
        print("\n=== Label coverage (non-null) ===")
        for lab in self.cfg.labels:
            n = agg_df[lab].notna().sum()
            print(f"{lab:<55s}: {n / total:6.2%} ({n}/{total})")
        print("=================================\n")

        out_path = os.path.join(self.cfg.save_dir, f"{base_name}_cleaned.csv")
        if self.cfg.differentiate:
            result = df_proc.merge(
                agg_df,
                left_on=[circle_column_name, square_column_name],
                right_index=True,
                how="left",
            )
        else:
            result = df_proc.merge(
                agg_df, left_on=column_name, right_index=True, how="left"
            )

        label_cols = list(self.cfg.labels.keys())

        if not self.cfg.differentiate and column_name in result.columns:
            cols = result.columns.tolist()
            cols.remove(column_name)
            cols.insert(0, column_name)
            result = result[cols]

        result.insert(1, "predicted_classes", result[label_cols].apply(_collect_predictions, axis=1))

        result_to_save = result.copy()
        result_to_save["predicted_classes"] = result_to_save["predicted_classes"].apply(json.dumps)
        result_to_save.to_csv(out_path, index=False)

        # keep raw response files for reference

        return result
