"""Econometric debiasing utilities for GABRIEL."""

from __future__ import annotations

import json
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
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


def _fit_ols_fallback(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool = True,  # noqa: ARG001 - signature parity
    varnames: Optional[List[str]] = None,  # noqa: ARG001
) -> Dict[str, Any]:
    """Minimal OLS routine used when statsmodels-backed fitting is unavailable."""

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
    except np.linalg.LinAlgError:
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
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid > 0 else np.nan

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


def _safe_fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool,
    varnames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Attempt statsmodels-backed fitting, falling back to NumPy OLS when needed."""

    try:
        return fit_ols(y, X, robust=robust, varnames=varnames)
    except ImportError:
        if robust:
            warnings.warn(
                "statsmodels is unavailable; falling back to non-robust OLS for debiasing diagnostics.",
                RuntimeWarning,
                stacklevel=2,
            )
        return _fit_ols_fallback(y, X, robust=False, varnames=varnames)

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
    attenuation_pct: Optional[float] = None
    diff_regression: Optional[Dict[str, Any]] = None
    twostep_regression: Optional[Dict[str, Any]] = None
    stage1_regression: Optional[Dict[str, Any]] = None
    stage1_delta: Optional[float] = None
    debiased_columns: Dict[str, str] = field(default_factory=dict)

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
            "attenuation_pct": _convert(self.attenuation_pct),
            "diff_regression": _convert(self.diff_regression),
            "twostep_regression": _convert(self.twostep_regression),
            "stage1_regression": _convert(self.stage1_regression),
            "stage1_delta": _convert(self.stage1_delta),
            "debiased_columns": _convert(self.debiased_columns),
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
    remaining_signal_attribute: Optional[str] = None
    remaining_signal_description: Optional[str] = None
    save_dir: str = DEFAULT_SAVE_DIR
    run_name: Optional[str] = None
    strip_percentages: Optional[List[int]] = None
    categories_to_strip: Optional[List[str]] = None
    template_path: Optional[str] = None
    model: str = "gpt-5-mini"
    n_parallels: int = 650
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
        attr_keys = list(self.cfg.attributes.keys())
        first_attr = attr_keys[0]
        measurement_attr = self.cfg.measurement_attribute
        if measurement_attr is not None and measurement_attr not in self.cfg.attributes:
            raise ValueError(
                f"Measurement attribute '{measurement_attr}' must be a key in attributes"
            )
        if measurement_attr is None:
            measurement_attr = first_attr
            if self.cfg.verbose:
                msg = (
                    "[Debias] measurement_attribute not provided; "
                    f"defaulting to '{measurement_attr}'."
                )
                if len(attr_keys) > 1:
                    msg += " Debiasing will use the first attribute provided."
                print(msg)
        elif len(attr_keys) > 1 and measurement_attr != first_attr:
            if self.cfg.verbose:
                print(
                    "[Debias] Multiple measurement attributes supplied; debiasing "
                    f"will use the first attribute '{first_attr}' and ignore the "
                    f"requested '{measurement_attr}'."
                )
            measurement_attr = first_attr
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

        if self.cfg.remaining_signal_attribute and not self.cfg.remaining_signal_description:
            raise ValueError(
                "remaining_signal_description must be provided when remaining_signal_attribute is set"
            )

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
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
    ) -> DebiasResult:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")

        df_master = df.copy().reset_index(drop=True)
        row_index = pd.RangeIndex(start=0, stop=len(df_master), step=1, name="__debias_row_id")
        df_master.index = row_index

        if self.cfg.verbose:
            print(f"[Debias] Running debiasing pipeline on {len(df_master)} rows.")

        attr_keys = list(self.cfg.attributes.keys())
        if self.cfg.verbose and len(attr_keys) > 1:
            print(
                "[Debias] Multiple attributes detected; debiasing will focus on "
                f"'{self.cfg.measurement_attribute}' while measuring all attributes "
                "on both raw and stripped text."
            )

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
            reset_files=reset_files,
        )
        self._attach_measurement(
            df_master,
            base_measure,
            attr_keys,
            variant_key="original",
            display_name="original",
        )

        if self.cfg.removal_method == "codify":
            variant_info = await self._prepare_codify_variants(
                df_master, column_name, reset_files=reset_files
            )
        else:
            variant_info = await self._prepare_paraphrase_variant(
                df_master, column_name, reset_files=reset_files
            )

        if variant_info and self.cfg.verbose:
            print("[Debias] Measuring stripped variants...")

        variant_count = len(variant_info)
        for info in variant_info.values():
            info["label_suffix"] = self._variant_label_suffix(info, variant_count)

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
                reset_files=reset_files,
            )
            column_map = self._attach_measurement(
                df_master,
                measure_df,
                attr_keys,
                variant_key=key,
                display_name=info["label_suffix"],
            )
            info["measurement_columns"] = column_map

        regression_info: Dict[str, DebiasRegressionResult] = {}
        for key, info in variant_info.items():
            stripped_column = info["measurement_columns"].get(
                self.cfg.measurement_attribute
            )
            if not stripped_column:
                continue
            remaining_signal_column = await self._measure_remaining_signal(
                df_master,
                info=info,
                save_label=key,
                label_suffix=info["label_suffix"],
                reset_files=reset_files,
            )
            summary = self._run_debiasing(
                df_master,
                original_column=self.cfg.measurement_attribute,
                stripped_column=stripped_column,
                remaining_signal_column=remaining_signal_column,
                variant_key=key,
                display_name=info["display"],
                strip_percentage=info.get("strip_percentage"),
                label_suffix=info["label_suffix"],
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
        reset_files: bool,
    ) -> pd.DataFrame:
        kwargs = dict(extra_kwargs or {})
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, save_label))
        os.makedirs(save_dir, exist_ok=True)
        reset_files_local = bool(kwargs.pop("reset_files", False)) or bool(reset_files)
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
            result = await runner.run(df_reset, column_name, reset_files=reset_files_local)
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
            result = await runner.run(df_reset, column_name, reset_files=reset_files_local)
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
            result = await runner.run(df_reset, column_name, reset_files=reset_files_local)
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
                reset_files=reset_files_local,
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

    # ------------------------------------------------------------------
    def _remaining_signal_attributes(self) -> Dict[str, str]:
        if not self.cfg.remaining_signal_attribute:
            return {}
        return {
            self.cfg.remaining_signal_attribute: str(self.cfg.remaining_signal_description)
        }

    # ------------------------------------------------------------------
    async def _measure_remaining_signal(
        self,
        df_master: pd.DataFrame,
        *,
        info: Dict[str, Any],
        save_label: str,
        label_suffix: str,
        reset_files: bool,
    ) -> Optional[str]:
        attrs = self._remaining_signal_attributes()
        if not attrs:
            return None
        if self.cfg.verbose:
            print(
                "[Debias] Measuring remaining-signal prevalence on "
                f"variant '{info['display']}'."
            )
        remaining_df = await self._run_measurement(
            df_master,
            column_name=info["text_column"],
            mode=self.cfg.mode,
            save_label=f"{save_label}_remaining_signal",
            attributes=attrs,
            template_path=self.cfg.template_path,
            extra_kwargs=self.cfg.measurement_kwargs,
            default_model=self.cfg.model,
            reset_files=reset_files,
        )

        measurement_attr = self.cfg.remaining_signal_attribute
        assert measurement_attr is not None
        prevalence_label = f"prevalence of {measurement_attr} ({label_suffix})"
        column_map = self._attach_measurement(
            df_master,
            remaining_df,
            [measurement_attr],
            variant_key=f"{save_label}_remaining_signal",
            display_name=prevalence_label,
        )
        return column_map.get(measurement_attr)

    # ------------------------------------------------------------------
    def _variant_label_suffix(self, info: Dict[str, Any], variant_count: int) -> str:
        pct = info.get("strip_percentage")
        display = str(info.get("display", "stripped"))
        if pct is None:
            if "paraphrase" in display.lower():
                return "stripped paraphrase"
            return "stripped"
        if pct == 100 and variant_count == 1:
            return "stripped"
        return f"stripped {pct}%"

    async def _prepare_codify_variants(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool,
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
            reset_files=reset_files,
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
        *,
        reset_files: bool,
    ) -> Dict[str, Dict[str, Any]]:
        kwargs = dict(self.cfg.removal_kwargs or {})
        save_dir = kwargs.pop("save_dir", os.path.join(self.run_dir, "paraphrase"))
        os.makedirs(save_dir, exist_ok=True)
        revised_name = f"{column_name} ({self.cfg.removal_attribute} stripped paraphrase)"
        instructions = kwargs.pop("instructions", None) or self._build_paraphrase_instructions()
        response_kwargs: Dict[str, Any] = {}
        if "n_rounds" in kwargs:
            response_kwargs["n_rounds"] = kwargs.pop("n_rounds")
        if "completion_max_rounds" in kwargs and "n_rounds" not in response_kwargs:
            replacement = kwargs.pop("completion_max_rounds")
            warnings.warn(
                "completion_max_rounds is deprecated; use n_rounds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if replacement is not None:
                response_kwargs["n_rounds"] = replacement
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
        paraphrased = await runner.run(
            df.reset_index(),
            column_name,
            reset_files=reset_files,
            **response_kwargs,
        )
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
            "Rewrite the passage so that every reference to the following signal(s) is removed while preserving all other content:",
        ]
        for key, desc in self.cfg.signal_dictionary.items():
            lines.append(f"- {key}: {desc}")
        lines.append(
            "Keep the remaining content identical. Be exhaustive: the final output must have absolutely no content which references, mentions, manifests, or alludes to the aformentioned signal(s). All such content must be stripped from the text; ensure nothing at all relevant remains. When in doubt, remove it. The stripping must be perfect; don't output anything related to / manifesting the signal(s). Your output is the original text, with everything else unchanged, but all this content completely and entirely removed."
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
    def _run_debiasing(
        self,
        df: pd.DataFrame,
        original_column: str,
        stripped_column: str,
        *,
        remaining_signal_column: Optional[str],
        variant_key: str,
        display_name: str,
        strip_percentage: Optional[int],
        label_suffix: str,
    ) -> DebiasRegressionResult:
        cols = [original_column, stripped_column]
        if remaining_signal_column:
            cols.append(remaining_signal_column)
        measurement_attr = self.cfg.measurement_attribute
        reg_df = df[cols].apply(pd.to_numeric, errors="coerce")
        reg_df = reg_df.dropna(subset=[original_column, stripped_column])
        if len(reg_df) < 3:
            if self.cfg.verbose:
                print(
                    f"[Debias] Not enough observations for debiasing on variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=None,
                mean_stripped=None,
            )

        y_series = reg_df[original_column]
        s_series = reg_df[stripped_column]

        if np.allclose(np.var(s_series.values), 0.0):
            if self.cfg.verbose:
                print(
                    f"[Debias] Stripped column has no variation for variant '{display_name}'."
                )
            return DebiasRegressionResult(
                variant=variant_key,
                display_name=display_name,
                strip_percentage=strip_percentage,
                correlation=None,
                mean_original=float(y_series.mean()),
                mean_stripped=float(s_series.mean()),
            )

        attenuation_pct = self._attenuation_pct(y_series, s_series)
        if self.cfg.verbose and attenuation_pct is not None:
            print(
                "[Debias] Mean attenuation from original to stripped "
                f"for '{display_name}': {attenuation_pct:.2f}%"
            )

        diff_col = f"{measurement_attr}__debiased_diff_{variant_key}"
        df.loc[reg_df.index, diff_col] = y_series - s_series
        pretty_diff_col = f"{measurement_attr} (raw - {label_suffix})"
        df.loc[reg_df.index, pretty_diff_col] = df.loc[reg_df.index, diff_col]

        mean_pct_diff, mean_abs_pct_diff = self._avg_pct_diff(
            y_series,
            df.loc[reg_df.index, diff_col],
        )
        if self.cfg.verbose:
            print(
                "[Debias] Simple difference debiasing percent change "
                f"(signed / absolute): {mean_pct_diff:.2f}% / {mean_abs_pct_diff:.2f}%"
            )

        diff_reg = self._regress_against_original(
            df,
            original_column=original_column,
            debiased_column=diff_col,
            title=f"Debiasing regression: original ~ (y - stripped) [{display_name}]",
            plot_filename=f"{variant_key}_diff_vs_original.png",
        )

        twostep_col = f"{measurement_attr}__debiased_twostep_{variant_key}"
        pretty_twostep_col = (
            f"{measurement_attr} (debiased)"
            if label_suffix == "stripped"
            else f"{measurement_attr} (debiased, {label_suffix})"
        )
        stage1_reg: Optional[Dict[str, Any]] = None
        delta: Optional[float] = None
        if remaining_signal_column and remaining_signal_column in df.columns:
            stage1_reg, delta = self._stage1_remaining_signal_regression(
                df,
                stripped_column=stripped_column,
                remaining_signal_column=remaining_signal_column,
                display_name=display_name,
            )
            if delta is not None:
                valid_idx, y_vals, s_vals, r_vals = self._aligned_series(
                    df,
                    original_column,
                    stripped_column,
                    remaining_signal_column,
                )
                df.loc[valid_idx, twostep_col] = y_vals - s_vals + delta * r_vals
                df.loc[valid_idx, pretty_twostep_col] = df.loc[valid_idx, twostep_col]
                mean_pct_two, mean_abs_pct_two = self._avg_pct_diff(
                    y_vals,
                    df.loc[valid_idx, twostep_col],
                )
                if self.cfg.verbose:
                    print(
                        "[Debias] Two-step debiasing percent change "
                        f"(signed / absolute): {mean_pct_two:.2f}% / {mean_abs_pct_two:.2f}%"
                    )
        else:
            if self.cfg.verbose and self.cfg.remaining_signal_attribute:
                print(
                    "[Debias] Remaining-signal attribute configured but no remaining-signal "
                    f"column was available for variant '{display_name}'."
                )

        twostep_reg: Optional[Dict[str, Any]] = None
        if twostep_col in df.columns and df[twostep_col].notna().sum() >= 3:
            twostep_reg = self._regress_against_original(
                df,
                original_column=original_column,
                debiased_column=twostep_col,
                title=f"Debiasing regression: original ~ two-step [{display_name}]",
                plot_filename=f"{variant_key}_twostep_vs_original.png",
            )

        correlation = float(y_series.corr(s_series))
        summary = DebiasRegressionResult(
            variant=variant_key,
            display_name=display_name,
            strip_percentage=strip_percentage,
            correlation=correlation,
            mean_original=float(y_series.mean()),
            mean_stripped=float(s_series.mean()),
            attenuation_pct=attenuation_pct,
            diff_regression=diff_reg,
            twostep_regression=twostep_reg,
            stage1_regression=stage1_reg,
            stage1_delta=delta,
            debiased_columns={
                "diff": pretty_diff_col,
                "twostep": pretty_twostep_col,
                "remaining_signal": remaining_signal_column or "",
            },
        )
        return summary

    # ------------------------------------------------------------------
    def _aligned_series(
        self,
        df: pd.DataFrame,
        original_column: str,
        stripped_column: str,
        remaining_signal_column: str,
    ) -> Tuple[pd.Index, pd.Series, pd.Series, pd.Series]:
        tmp = df[[original_column, stripped_column, remaining_signal_column]].apply(
            pd.to_numeric,
            errors="coerce",
        )
        tmp = tmp.dropna()
        return (
            tmp.index,
            tmp[original_column],
            tmp[stripped_column],
            tmp[remaining_signal_column],
        )

    # ------------------------------------------------------------------
    def _attenuation_pct(
        self,
        original: pd.Series,
        stripped: pd.Series,
    ) -> Optional[float]:
        mean_original = float(pd.to_numeric(original, errors="coerce").mean())
        mean_stripped = float(pd.to_numeric(stripped, errors="coerce").mean())
        if np.isclose(mean_original, 0.0):
            return None
        return float((mean_original - mean_stripped) / mean_original * 100.0)

    # ------------------------------------------------------------------
    def _avg_pct_diff(
        self,
        raw: pd.Series,
        debiased: pd.Series,
    ) -> Tuple[float, float]:
        raw_num = pd.to_numeric(raw, errors="coerce")
        deb_num = pd.to_numeric(debiased, errors="coerce")
        pct = (deb_num - raw_num) / raw_num * 100.0
        pct = pct.replace([np.inf, -np.inf], np.nan)
        pct = pct[raw_num != 0]
        return float(pct.mean()), float(pct.abs().mean())

    # ------------------------------------------------------------------
    def _stage1_remaining_signal_regression(
        self,
        df: pd.DataFrame,
        *,
        stripped_column: str,
        remaining_signal_column: str,
        display_name: str,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        stage1_df = df[[stripped_column, remaining_signal_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(stage1_df) < 3:
            if self.cfg.verbose:
                print(
                    "[Debias] Not enough observations for remaining-signal regression "
                    f"on variant '{display_name}'."
                )
            return None, None

        r_vals = stage1_df[remaining_signal_column].values
        if np.allclose(np.var(r_vals), 0.0):
            if self.cfg.verbose:
                print(
                    "[Debias] Remaining-signal proxy has no variation for "
                    f"variant '{display_name}'."
                )
            return None, None

        s_vals = stage1_df[stripped_column].values
        X1 = np.column_stack([np.ones(len(stage1_df)), r_vals])
        reg = _safe_fit_ols(
            s_vals,
            X1,
            robust=self.cfg.robust_regression,
            varnames=["Intercept", remaining_signal_column],
        )
        if self.cfg.verbose:
            self._print_generic_regression_table(
                reg,
                names=["Intercept", remaining_signal_column],
                title=f"Stage-1 regression: stripped ~ remaining signal [{display_name}]",
            )
        delta = float(reg["coef"][1])
        return self._regression_dict(reg, ["Intercept", remaining_signal_column]), delta

    # ------------------------------------------------------------------
    def _regress_against_original(
        self,
        df: pd.DataFrame,
        *,
        original_column: str,
        debiased_column: str,
        title: str,
        plot_filename: str,
    ) -> Dict[str, Any]:
        reg_df = df[[original_column, debiased_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(reg_df) < 3:
            return {}
        y_vals = reg_df[original_column].values
        x_vals = reg_df[debiased_column].values
        X = np.column_stack([np.ones(len(reg_df)), x_vals])
        reg_res = _safe_fit_ols(
            y_vals,
            X,
            robust=self.cfg.robust_regression,
            varnames=["Intercept", debiased_column],
        )
        if self.cfg.verbose:
            self._print_generic_regression_table(
                reg_res,
                names=["Intercept", debiased_column],
                title=title,
            )
        self._save_scatter_with_fit(
            reg_df,
            x_col=debiased_column,
            y_col=original_column,
            reg_res=reg_res,
            filename=plot_filename,
            title=title,
        )
        return self._regression_dict(reg_res, ["Intercept", debiased_column]) or {}

    # ------------------------------------------------------------------
    def _save_scatter_with_fit(
        self,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        reg_res: Dict[str, Any],
        filename: str,
        title: str,
    ) -> Optional[str]:
        if df.empty:
            return None
        x_vals = df[x_col].values.astype(float)
        y_vals = df[y_col].values.astype(float)
        beta0, beta1 = reg_res["coef"][0], reg_res["coef"][1]
        x_line = np.linspace(float(np.min(x_vals)), float(np.max(x_vals)), 200)
        y_line = beta0 + beta1 * x_line

        fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
        ax.scatter(x_vals, y_vals, alpha=0.35, edgecolor="none")
        ax.plot(x_line, y_line, color="black", linewidth=2, label="OLS fit")
        ax.set_title(title)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        plot_path = os.path.join(self.run_dir, filename)
        fig.savefig(plot_path)
        plt.close(fig)
        if self.cfg.verbose:
            print(f"[Debias] Saved regression plot to {plot_path}")
        return plot_path

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
    def _print_generic_regression_table(
        self,
        result: Dict[str, Any],
        *,
        names: List[str],
        title: str,
    ) -> None:
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
