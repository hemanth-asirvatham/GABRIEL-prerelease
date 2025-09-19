"""Econometric debiasing utilities for GABRIEL."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - tqdm is optional
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: Any) -> Iterable:
        return iterable

from .classify import Classify, ClassifyConfig
from .codify import Codify, CodifyConfig
from .extract import Extract, ExtractConfig
from .paraphrase import Paraphrase, ParaphraseConfig
from .rank import Rank, RankConfig
from .rate import Rate, RateConfig
try:  # statsmodels is optional; fall back to a lightweight solver if missing
    from ..utils.plot_utils import fit_ols as _fit_ols
except Exception:  # pragma: no cover - fallback exercised when statsmodels absent

    def fit_ols(
        y: np.ndarray,
        X: np.ndarray,
        *,
        robust: bool = True,  # noqa: ARG001 - signature parity with primary implementation
        varnames: Optional[List[str]] = None,  # noqa: ARG001
    ) -> Dict[str, Any]:
        """Minimal OLS routine used when :mod:`statsmodels` is unavailable.

        The implementation mirrors the API of
        :func:`gabriel.utils.plot_utils.fit_ols` closely enough for the
        debiasing pipeline, returning coefficient estimates, approximate
        standard errors, and residuals.  ``robust`` and ``varnames`` are
        accepted for signature compatibility but otherwise ignored.
        """

        y_arr = np.asarray(y, dtype=float)
        X_arr = np.asarray(X, dtype=float)
        if y_arr.ndim != 1:
            y_arr = y_arr.reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError("Design matrix must be two-dimensional")

        beta, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)
        resid = y_arr - X_arr @ beta
        n, k_plus1 = X_arr.shape
        df_resid = n - k_plus1

        XtX = X_arr.T @ X_arr
        try:
            XtX_inv = np.linalg.inv(XtX)
        except np.linalg.LinAlgError:  # pseudo-inverse when design is singular
            XtX_inv = np.linalg.pinv(XtX)

        if df_resid > 0:
            sigma2 = float(resid @ resid) / df_resid
            cov = sigma2 * XtX_inv
            se = np.sqrt(np.diag(cov))
            rse = float(np.sqrt(sigma2))
        else:
            sigma2 = 0.0
            se = np.full(beta.shape, np.nan)
            rse = np.nan

        with np.errstate(divide="ignore", invalid="ignore"):
            t_vals = beta / se
        p_vals = np.full(beta.shape, np.nan)

        mean_y = float(y_arr.mean()) if n else 0.0
        ss_tot = float(np.sum((y_arr - mean_y) ** 2))
        ss_res = float(np.sum(resid ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        adj_r2 = (
            1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else np.nan
        )

        return {
            "coef": np.asarray(beta),
            "se": np.asarray(se),
            "t": np.asarray(t_vals),
            "p": np.asarray(p_vals),
            "r2": float(r2),
            "adj_r2": float(adj_r2),
            "n": int(n),
            "k": int(k_plus1 - 1),
            "rse": rse,
            "F": np.nan,
            "resid": np.asarray(resid),
            "varnames": varnames,
            "sm_results": None,
        }
else:  # pragma: no cover - executed when statsmodels dependency is available
    fit_ols = _fit_ols

DEFAULT_SAVE_DIR = os.path.expanduser("~/Documents/runs")
RemovalMethod = Literal["codify", "paraphrase"]
MeasurementMode = Literal["rate", "classify", "extract", "rank"]


@dataclass
class DebiasRegressionResult:
    """Container for regression diagnostics of a debiasing run."""

    variant: str
    display_name: str
    strip_percentage: Optional[int]
    correlation: Optional[float]
    mean_original: Optional[float]
    mean_stripped: Optional[float]
    regression: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON serialisable representation of the result."""

        def _convert(value: Any) -> Any:
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            if isinstance(value, np.generic):
                return value.item()
            return value

        return {
            "variant": self.variant,
            "display_name": self.display_name,
            "strip_percentage": self.strip_percentage,
            "correlation": _convert(self.correlation),
            "mean_original": _convert(self.mean_original),
            "mean_stripped": _convert(self.mean_stripped),
            "regression": _convert(self.regression),
        }


@dataclass
class DebiasConfig:
    """Configuration for :class:`DebiasPipeline`."""

    mode: MeasurementMode = "rate"
    measurement_attribute: Optional[str] = None
    removal_attribute: Optional[str] = None
    attributes: Dict[str, str] = field(default_factory=dict)
    signal_dictionary: Dict[str, str] = field(default_factory=dict)
    removal_method: RemovalMethod = "codify"
    save_dir: str = DEFAULT_SAVE_DIR
    run_name: Optional[str] = None
    strip_percentages: Optional[List[int]] = None
    categories_to_strip: Optional[List[str]] = None
    template_path: Optional[str] = None
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    measurement_kwargs: Dict[str, Any] = field(default_factory=dict)
    removal_kwargs: Dict[str, Any] = field(default_factory=dict)
    use_dummy: bool = False
    robust_regression: bool = True
    random_seed: int = 12345
    verbose: bool = True


@dataclass
class DebiasResult:
    """Return object for :func:`gabriel.debias`."""

    results: pd.DataFrame
    metadata: Dict[str, Any]
    regression: Dict[str, DebiasRegressionResult]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the result metadata for convenient inspection."""

        return {
            "metadata": self.metadata,
            "regression": {k: v.as_dict() for k, v in self.regression.items()},
        }


class DebiasPipeline:
    """Coordinate debiasing runs that rely on core Gabriel tasks."""

    def __init__(self, config: DebiasConfig) -> None:
        self.cfg = config
        self._validate_config()
        base_dir = os.path.expandvars(os.path.expanduser(self.cfg.save_dir))
        os.makedirs(base_dir, exist_ok=True)
        run_name = self.cfg.run_name or self._default_run_name()
        self.run_dir = os.path.join(base_dir, run_name)
        os.makedirs(self.run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _default_run_name(self) -> str:
        base_name = (
            self.cfg.measurement_attribute
            or self.cfg.removal_attribute
            or "signal"
        )
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", base_name)
        prefix = "debias"
        cleaned = cleaned.strip("_")
        if cleaned:
            return f"{prefix}_{cleaned}"
        return prefix

    # ------------------------------------------------------------------
    def _validate_config(self) -> None:
        valid_modes = {"rate", "classify", "extract", "rank"}
        if self.cfg.mode not in valid_modes:
            raise ValueError("mode must be one of {'rate', 'classify', 'extract', 'rank'}")
        self.cfg.attributes = dict(self.cfg.attributes or {})
        if not self.cfg.attributes:
            raise ValueError("attributes must be supplied for the selected mode")

        self.cfg.signal_dictionary = dict(self.cfg.signal_dictionary or {})
        if not self.cfg.signal_dictionary:
            raise ValueError("signal_dictionary must describe the signal to remove")
        measurement_attr = self.cfg.measurement_attribute
        if measurement_attr is None:
            measurement_attr = next(iter(self.cfg.attributes))
            if self.cfg.verbose:
                print(
                    "[Debias] measurement_attribute not provided; "
                    f"defaulting to '{measurement_attr}'."
                )
        elif measurement_attr not in self.cfg.attributes:
            raise ValueError(
                f"Measurement attribute '{measurement_attr}' must be a key in attributes"
            )
        self.cfg.measurement_attribute = measurement_attr

        removal_attr = self.cfg.removal_attribute
        if removal_attr is None:
            if measurement_attr in self.cfg.signal_dictionary:
                removal_attr = measurement_attr
                if self.cfg.verbose:
                    print(
                        "[Debias] removal_attribute not provided; "
                        f"defaulting to measurement attribute '{removal_attr}'."
                    )
            else:
                removal_attr = next(iter(self.cfg.signal_dictionary))
                if self.cfg.verbose:
                    print(
                        "[Debias] removal_attribute not provided; "
                        f"defaulting to '{removal_attr}'."
                    )
        elif removal_attr not in self.cfg.signal_dictionary:
            raise ValueError(
                f"Removal attribute '{removal_attr}' must be a key in signal_dictionary"
            )
        self.cfg.removal_attribute = removal_attr
        if self.cfg.removal_method not in {"codify", "paraphrase"}:
            raise ValueError("removal_method must be 'codify' or 'paraphrase'")

        if self.cfg.categories_to_strip is None:
            if self.cfg.removal_method == "codify":
                self.cfg.categories_to_strip = [self.cfg.removal_attribute]
            else:
                self.cfg.categories_to_strip = []
        else:
            categories: List[str] = []
            for name in self.cfg.categories_to_strip:
                if name in self.cfg.signal_dictionary and name not in categories:
                    categories.append(name)
            if self.cfg.removal_method == "codify" and not categories:
                raise ValueError(
                    "categories_to_strip must contain at least one key from signal_dictionary when using codify"
                )
            self.cfg.categories_to_strip = categories

        if self.cfg.strip_percentages is None:
            percentages: List[int] = [100]
        else:
            percentages = []
            for pct in self.cfg.strip_percentages:
                try:
                    val = int(pct)
                except Exception as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid strip percentage: {pct!r}") from exc
                val = max(0, min(100, val))
                if val not in percentages:
                    percentages.append(val)
            if not percentages:
                percentages = [100]
        percentages.sort()
        self.cfg.strip_percentages = percentages

        self.cfg.measurement_kwargs = dict(self.cfg.measurement_kwargs or {})
        self.cfg.removal_kwargs = dict(self.cfg.removal_kwargs or {})

    # ------------------------------------------------------------------
    async def run(self, df: pd.DataFrame, column_name: str) -> DebiasResult:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        df_master = df.copy().reset_index(drop=True)
        row_index = pd.RangeIndex(start=0, stop=len(df_master), step=1, name="__debias_row_id")
        df_master.index = row_index

        if self.cfg.verbose:
            print(f"[Debias] Running debiasing pipeline on {len(df_master)} rows.")

        attr_keys = list(self.cfg.attributes.keys())

        if self.cfg.verbose:
            print("[Debias] Measuring baseline signals...")
        base_measure = await self._run_measurement(
            df_master,
            column_name=column_name,
            mode=self.cfg.mode,
            save_label="original",
            attributes=self.cfg.attributes,
            template_path=self.cfg.template_path,
            extra_kwargs=self.cfg.measurement_kwargs,
            default_model=self.cfg.model,
        )
        self._attach_measurement(
            df_master,
            base_measure,
            attr_keys,
            variant_key="original",
            display_name="original",
        )

        if self.cfg.removal_method == "codify":
            variant_info = await self._prepare_codify_variants(df_master, column_name)
        else:
            variant_info = await self._prepare_paraphrase_variant(df_master, column_name)

        if variant_info and self.cfg.verbose:
            print("[Debias] Measuring stripped variants...")

        disable_progress = not (self.cfg.verbose and bool(variant_info))
        for key in tqdm(list(variant_info.keys()), desc="Variants", disable=disable_progress):
            info = variant_info[key]
            measure_df = await self._run_measurement(
                df_master,
                column_name=info["text_column"],
                mode=self.cfg.mode,
                save_label=key,
                attributes=self.cfg.attributes,
                template_path=self.cfg.template_path,
                extra_kwargs=self.cfg.measurement_kwargs,
                default_model=self.cfg.model,
            )
            column_map = self._attach_measurement(
                df_master,
                measure_df,
                attr_keys,
                variant_key=key,
                display_name=info["display"],
            )
            info["measurement_columns"] = column_map

        regression_info: Dict[str, DebiasRegressionResult] = {}
        for key, info in variant_info.items():
            stripped_column = info["measurement_columns"].get(
                self.cfg.measurement_attribute
            )
            if not stripped_column:
                continue
            summary = self._run_regression(
                df_master,
                original_column=self.cfg.measurement_attribute,
                stripped_column=stripped_column,
                variant_key=key,
                display_name=info["display"],
                strip_percentage=info.get("strip_percentage"),
            )
            regression_info[key] = summary

        metadata = {
            "config": self._serialise_config(),
            "variants": [regression_info[k].as_dict() for k in regression_info],
            "result_path": os.path.join(self.run_dir, "debias_results.csv"),
        }
        metadata_path = os.path.join(self.run_dir, "debias_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        metadata["metadata_path"] = metadata_path

        results_df = df_master.reset_index(drop=True)
        results_path = metadata["result_path"]
        results_df.to_csv(results_path, index=False)
        if self.cfg.verbose:
            print(f"[Debias] Results saved to {results_path}")

        return DebiasResult(
            results=results_df,
            metadata=metadata,
            regression=regression_info,
        )

    # ------------------------------------------------------------------
    async def _run_measurement(
        self,
        df: pd.DataFrame,
        *,
        column_name: str,
        mode: MeasurementMode,
        save_label: str,
        attributes: Optional[Dict[str, str]],
        template_path: Optional[str],
        extra_kwargs: Optional[Dict[str, Any]],
        default_model: str,
    ) -> pd.DataFrame:
        kwargs = dict(extra_kwargs or {})
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, save_label))
        os.makedirs(save_dir, exist_ok=True)
        reset_files = bool(kwargs.pop("reset_files", False))
        df_reset = df.reset_index()
        if mode == "rate":
            cfg = RateConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
                **kwargs,
            )
            runner = Rate(cfg, template_path=template_path)
            result = await runner.run(df_reset, column_name, reset_files=reset_files)
        elif mode == "classify":
            cfg = ClassifyConfig(
                labels=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
                **kwargs,
            )
            runner = Classify(cfg, template_path=template_path)
            result = await runner.run(df_reset, column_name, reset_files=reset_files)
        elif mode == "extract":
            cfg = ExtractConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
                **kwargs,
            )
            runner = Extract(cfg, template_path=template_path)
            result = await runner.run(df_reset, column_name, reset_files=reset_files)
        elif mode == "rank":
            cfg = RankConfig(
                attributes=attributes or {},
                save_dir=save_dir,
                model=kwargs.pop("model", default_model),
                n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
                use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
                **kwargs,
            )
            runner = Rank(cfg, template_path=template_path)
            result = await runner.run(
                df_reset,
                column_name,
                id_column="__debias_row_id",
                reset_files=reset_files,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported mode: {mode}")

        if "__debias_row_id" in result.columns:
            result = result.set_index("__debias_row_id")
        elif "identifier" in result.columns:
            result = result.set_index("identifier")
        else:
            result.index = df.index
        result.index.name = "__debias_row_id"
        return result

    # ------------------------------------------------------------------
    def _attach_measurement(
        self,
        df_master: pd.DataFrame,
        measurement_df: pd.DataFrame,
        attributes: List[str],
        *,
        variant_key: str,
        display_name: str,
    ) -> Dict[str, str]:
        column_map: Dict[str, str] = {}
        for attr in attributes:
            if attr not in measurement_df.columns:
                continue
            if variant_key == "original":
                target_name = attr
            else:
                target_name = f"{attr} ({display_name})"
            df_master[target_name] = measurement_df[attr].reindex(df_master.index)
            column_map[attr] = target_name
        return column_map

    async def _prepare_codify_variants(
        self,
        df: pd.DataFrame,
        column_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        kwargs = dict(self.cfg.removal_kwargs or {})
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, "codify"))
        os.makedirs(save_dir, exist_ok=True)
        additional_instructions = kwargs.pop("additional_instructions", "")
        cfg = CodifyConfig(
            save_dir=save_dir,
            model=kwargs.pop("model", self.cfg.model),
            n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
            use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
            **kwargs,
        )
        runner = Codify(cfg)
        codify_df = await runner.run(
            df.reset_index(),
            column_name,
            categories=self.cfg.signal_dictionary,
            additional_instructions=additional_instructions,
        )
        variants: Dict[str, Dict[str, Any]] = {}
        categories = self.cfg.categories_to_strip or []
        for pct in self.cfg.strip_percentages:
            if pct <= 0:
                continue
            key = f"stripped_{pct:03d}pct"
            display = f"{self.cfg.removal_attribute} stripped {pct}%"
            new_col = f"{column_name} ({display})"
            df[new_col] = [
                self._strip_passages(
                    original_text=str(df.at[idx, column_name]),
                    snippets=self._collect_snippets(codify_df, idx, categories),
                    pct_strip=pct,
                    row_idx=int(idx),
                )
                for idx in df.index
            ]
            variants[key] = {
                "text_column": new_col,
                "display": display,
                "strip_percentage": pct,
            }
        return variants

    # ------------------------------------------------------------------
    async def _prepare_paraphrase_variant(
        self,
        df: pd.DataFrame,
        column_name: str,
    ) -> Dict[str, Dict[str, Any]]:
        kwargs = dict(self.cfg.removal_kwargs or {})
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, "paraphrase"))
        os.makedirs(save_dir, exist_ok=True)
        revised_name = f"{column_name} ({self.cfg.removal_attribute} stripped paraphrase)"
        instructions = kwargs.pop("instructions", None) or self._build_paraphrase_instructions()
        response_kwargs: Dict[str, Any] = {}
        if "completion_max_rounds" in kwargs:
            response_kwargs["completion_max_rounds"] = kwargs.pop("completion_max_rounds")
        cfg = ParaphraseConfig(
            instructions=instructions,
            revised_column_name=revised_name,
            save_dir=save_dir,
            model=kwargs.pop("model", self.cfg.model),
            n_parallels=kwargs.pop("n_parallels", self.cfg.n_parallels),
            use_dummy=kwargs.pop("use_dummy", self.cfg.use_dummy),
            **kwargs,
        )
        runner = Paraphrase(cfg)
        paraphrased = await runner.run(df.reset_index(), column_name, **response_kwargs)
        df[revised_name] = paraphrased[revised_name].reindex(df.index)
        return {
            "paraphrase": {
                "text_column": revised_name,
                "display": f"{self.cfg.removal_attribute} stripped (paraphrase)",
                "strip_percentage": None,
            }
        }

    # ------------------------------------------------------------------
    def _build_paraphrase_instructions(self) -> str:
        lines = [
            "Rewrite the passage so that every reference to the following signals is removed while preserving all other content:",
        ]
        for key, desc in self.cfg.signal_dictionary.items():
            lines.append(f"- {key}: {desc}")
        lines.append(
            "Keep the remaining wording, ordering, and style identical unless necessary for grammatical correctness."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _collect_snippets(
        self,
        codify_df: pd.DataFrame,
        row_idx: int,
        categories: List[str],
    ) -> List[str]:
        snippets: List[str] = []
        for cat in categories:
            if cat not in codify_df.columns:
                continue
            raw = codify_df.at[row_idx, cat]
            if isinstance(raw, list):
                snippets.extend(str(s) for s in raw if s)
            elif isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        snippets.extend(str(s) for s in parsed if s)
                except Exception:
                    if raw:
                        snippets.append(str(raw))
        return snippets

    # ------------------------------------------------------------------
    def _strip_passages(
        self,
        *,
        original_text: str,
        snippets: List[str],
        pct_strip: int,
        row_idx: int,
    ) -> str:
        if pct_strip <= 0 or not snippets:
            return self._normalise_ws(original_text)
        unique: List[str] = []
        seen = set()
        for snippet in snippets:
            snippet = snippet.strip()
            if not snippet:
                continue
            if snippet not in seen:
                seen.add(snippet)
                unique.append(snippet)
        if not unique:
            return self._normalise_ws(original_text)
        n_total = len(unique)
        n_remove = max(0, min(n_total, int(round(n_total * (pct_strip / 100.0)))))
        if n_remove == 0:
            return self._normalise_ws(original_text)
        seed_val = f"{self.cfg.random_seed}:{row_idx}:{pct_strip}:{n_total}"
        rng = random.Random()
        rng.seed(seed_val)
        to_remove = rng.sample(unique, n_remove)
        cleaned = original_text
        for snippet in sorted(to_remove, key=len, reverse=True):
            cleaned = cleaned.replace(snippet, " ")
        return self._normalise_ws(cleaned)

    # ------------------------------------------------------------------
    def _run_regression(
        self,
        df: pd.DataFrame,
        original_column: str,
        stripped_column: str,
        *,
        variant_key: str,
        display_name: str,
        strip_percentage: Optional[int],
    ) -> DebiasRegressionResult:
        cols = [original_column, stripped_column]
        measurement_attr = self.cfg.measurement_attribute
        rename_map = {
            original_column: f"{measurement_attr} (original)",
            stripped_column: f"{measurement_attr} ({display_name})",
        }
        reg_df = df[cols].apply(pd.to_numeric, errors="coerce")
        reg_df = reg_df.dropna()
        if len(reg_df) < 3:
            if self.cfg.verbose:
                print(
                    f"[Debias] Not enough observations for regression on variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=None,
                mean_stripped=None,
            )
        y = reg_df[original_column].values
        x = reg_df[stripped_column].values
        if np.allclose(np.var(x), 0.0):
            if self.cfg.verbose:
                print(
                    f"[Debias] Stripped column has no variation for variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=float(reg_df[original_column].mean()),
                mean_stripped=float(reg_df[stripped_column].mean()),
            )
        X = np.column_stack([np.ones(len(reg_df)), x])
        reg_res = fit_ols(
            y,
            X,
            robust=self.cfg.robust_regression,
            varnames=["Intercept", rename_map[stripped_column]],
        )
        if self.cfg.verbose:
            self._print_regression_table(
                reg_res,
                rename_map,
                stripped_column,
                title=f"Regression ({display_name})",
            )
        resid = pd.Series(
            reg_res["resid"],
            index=reg_df.index,
            name=f"{measurement_attr}__residual_{variant_key}",
        )
        df.loc[resid.index, resid.name] = resid
        df.loc[resid.index, f"{measurement_attr}__debiased_{variant_key}"] = resid
        correlation = float(reg_df[original_column].corr(reg_df[stripped_column]))
        summary = DebiasRegressionResult(
            variant=variant_key,
            display_name=display_name,
            strip_percentage=strip_percentage,
            correlation=correlation,
            mean_original=float(reg_df[original_column].mean()),
            mean_stripped=float(reg_df[stripped_column].mean()),
            regression=self._regression_dict(reg_res, ["Intercept", rename_map[stripped_column]]),
        )
        return summary

    # ------------------------------------------------------------------
    def _print_regression_table(
        self,
        result: Dict[str, Any],
        rename_map: Dict[str, str],
        stripped_column: str,
        title: str,
    ) -> None:
        names = ["Intercept", rename_map[stripped_column]]
        table = pd.DataFrame(
            {
                "coef": result["coef"],
                "se": result["se"],
                "t": result["t"],
                "p": result["p"],
            },
            index=names,
        )
        print(f"\n{title}")
        print(table.round(6).to_string())
        print(
            f"R^2 = {result['r2']:.4f}, adj. R^2 = {result['adj_r2']:.4f}, n = {result['n']}"
        )

    # ------------------------------------------------------------------
    def _regression_dict(
        self,
        result: Optional[Dict[str, Any]],
        names: List[str],
    ) -> Optional[Dict[str, Any]]:
        if result is None:
            return None
        return {
            "coef": {name: float(val) for name, val in zip(names, result["coef"])},
            "se": {name: float(val) for name, val in zip(names, result["se"])},
            "t": {name: float(val) for name, val in zip(names, result["t"])},
            "p": {name: float(val) for name, val in zip(names, result["p"])},
            "r2": float(result["r2"]),
            "adj_r2": float(result["adj_r2"]),
            "n": int(result["n"]),
        }

    # ------------------------------------------------------------------
    def _serialise_config(self) -> Dict[str, Any]:
        cfg_dict = asdict(self.cfg)
        cfg_dict["save_dir"] = os.path.expandvars(os.path.expanduser(self.cfg.save_dir))
        return cfg_dict

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_ws(text: str) -> str:
        return " ".join(str(text or "").split())
