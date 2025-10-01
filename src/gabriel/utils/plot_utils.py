"""
Enhanced Gabriel Visualisation Utilities
=======================================

This module refines the original plotting utilities to provide:

* OLS regressions via statsmodels with meaningful coefficient names
  (no more ``x1``, ``x2``) and optional robust standard errors.
* Binned scatter plots that support multiple independent variables via
  ``controls`` and allow custom axis limits.
* Bar, box and line plots with a variety of customisation options.

The functions mirror the earlier API but with cleaner parameter names
and additional features.  For Python 3.12 and SciPy 1.16+, use
``statsmodels>=0.14.5`` to avoid import errors.
"""

from __future__ import annotations

import textwrap
from typing import Iterable, Dict, Any, Optional, List, Tuple, Sequence, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from scipy.stats import sem

try:
    from tabulate import tabulate  # type: ignore
except ModuleNotFoundError:
    tabulate = None  # fallback when tabulate isn't installed

import statsmodels.api as sm
import statsmodels.formula.api as smf


def _ensure_list(values: Optional[Union[str, Sequence[str]]]) -> List[str]:
    """Return ``values`` as a list, accepting strings or iterables."""

    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return list(values)


def _cluster_groups(df: pd.DataFrame, columns: Sequence[str]) -> Union[np.ndarray, pd.Series]:
    """Return an array of cluster identifiers suitable for statsmodels."""

    if len(columns) == 1:
        col = columns[0]
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            return series.values
        return pd.Categorical(series).codes
    group_df = pd.DataFrame(index=df.index)
    for col in columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            group_df[col] = series.values
        else:
            group_df[col] = pd.Categorical(series).codes
    return group_df.values


def _apply_year_excess(
    df: pd.DataFrame,
    *,
    year_col: str,
    window: int,
    columns: Sequence[str],
    mode: str = "difference",
    replace: bool = True,
    prefix: str = "",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Compute excess/ratio values relative to a rolling window of years.

    Parameters
    ----------
    df : DataFrame
        Data containing the original variables and ``year_col``.
    year_col : str
        Column representing the temporal dimension.
    window : int
        Number of years on each side used when computing the rolling mean.
    columns : Sequence[str]
        Variables for which to compute excess or ratios.
    mode : {"difference", "ratio"}
        Whether to subtract (excess) or divide (ratio) by the window mean.
    replace : bool, default True
        If True, the returned mapping replaces each original column name with
        the derived excess/ratio column in downstream analyses.
    prefix : str, default ""
        Optional prefix for new columns (useful when running multiple configs).

    Returns
    -------
    df_out : DataFrame
        Copy of ``df`` containing the additional columns.
    replacements : dict
        Mapping of original column names to new excess/ratio columns that can
        be used to update variable lists.
    """

    if year_col not in df.columns:
        raise KeyError(f"Year column '{year_col}' not found in dataframe.")
    if mode not in {"difference", "ratio"}:
        raise ValueError("mode must be 'difference' or 'ratio'.")
    df_out = df.copy()
    missing = [col for col in columns if col not in df_out.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in dataframe for excess calculation.")
    df_out = df_out.sort_values(year_col)
    unique_years = df_out[year_col].dropna().unique()
    unique_years.sort()
    replacements: Dict[str, str] = {}
    means: Dict[str, Dict[Any, float]] = {col: {} for col in columns}
    for i, year in enumerate(unique_years):
        lower_idx = max(0, i - window)
        upper_idx = min(len(unique_years) - 1, i + window)
        relevant_years = unique_years[lower_idx : upper_idx + 1]
        subset = df_out[df_out[year_col].isin(relevant_years)]
        year_means = subset[columns].mean()
        for col in columns:
            means[col][year] = year_means.get(col, np.nan)
    for col in columns:
        mean_col = f"{prefix}{col}__year_mean"
        df_out[mean_col] = df_out[year_col].map(means[col])
        new_col = f"{prefix}{col}_{'excess' if mode == 'difference' else 'ratio'}"
        if mode == "difference":
            df_out[new_col] = df_out[col] - df_out[mean_col]
        else:
            df_out[new_col] = df_out[col] / df_out[mean_col]
        if replace:
            replacements[col] = new_col
    return df_out, replacements


def _format_coefficient(coef: float, se: Optional[float], pval: Optional[float], *, float_fmt: str) -> str:
    """Return a formatted coefficient string with standard error and stars."""

    if np.isnan(coef):
        return "-"
    stars = ""
    if pval is not None:
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.1:
            stars = "*"
    coef_part = float_fmt.format(coef)
    if se is None or np.isnan(se):
        return f"{coef_part}{stars}"
    se_part = float_fmt.format(se)
    return f"{coef_part} ({se_part}){stars}"


def _results_to_dict(
    res: sm.regression.linear_model.RegressionResultsWrapper,
    *,
    display_varnames: List[str],
    param_lookup: Dict[str, str],
) -> Dict[str, Any]:
    """Convert a statsmodels result object to the dictionary structure used here."""

    params = res.params
    se = res.bse
    return {
        "coef": params,
        "se": se,
        "t": res.tvalues,
        "p": res.pvalues,
        "r2": getattr(res, "rsquared", np.nan),
        "adj_r2": getattr(res, "rsquared_adj", np.nan),
        "n": int(res.nobs),
        "k": len(params) - 1 if "Intercept" in params.index else len(params),
        "rse": np.sqrt(res.mse_resid) if hasattr(res, "mse_resid") else np.nan,
        "F": getattr(res, "fvalue", np.nan),
        "resid": res.resid,
        "varnames": list(params.index),
        "display_varnames": display_varnames,
        "param_lookup": param_lookup,
        "sm_results": res,
    }


def _fit_formula_model(
    data: pd.DataFrame,
    *,
    y: str,
    main_vars: Sequence[str],
    main_display: Sequence[str],
    controls: Sequence[str],
    control_display: Sequence[str],
    robust: bool,
    entity_fe: Optional[str],
    time_fe: Optional[str],
    include_intercept: bool,
    cluster_cols: Sequence[str],
) -> Tuple[Dict[str, Any], str]:
    """Fit an OLS model via formulas, optionally with fixed effects."""

    rhs_terms = [f"Q('{var}')" for var in main_vars]
    rhs_terms.extend(f"Q('{var}')" for var in controls)
    if entity_fe:
        rhs_terms.append(f"C(Q('{entity_fe}'))")
    if time_fe:
        rhs_terms.append(f"C(Q('{time_fe}'))")
    if not rhs_terms:
        rhs_terms = ["1"]
    formula = f"Q('{y}') ~ " + " + ".join(rhs_terms)
    if not include_intercept:
        formula += " - 1"
    model = smf.ols(formula=formula, data=data)
    fit_kwargs: Dict[str, Any] = {}
    if cluster_cols:
        groups = _cluster_groups(data, cluster_cols)
        fit_kwargs["cov_type"] = "cluster"
        fit_kwargs["cov_kwds"] = {"groups": groups}
    elif robust:
        fit_kwargs["cov_type"] = "HC3"
    try:
        res = model.fit(**fit_kwargs)
    except ValueError:
        if fit_kwargs.get("cov_type") == "HC3":
            fit_kwargs["cov_type"] = "HC1"
            res = model.fit(**fit_kwargs)
        else:
            raise
    display_varnames: List[str] = []
    param_lookup: Dict[str, str] = {}
    if include_intercept and "Intercept" in res.params.index:
        display_varnames.append("Intercept")
        param_lookup["Intercept"] = "Intercept"
    for var, disp in zip(main_vars, main_display):
        key = f"Q('{var}')"
        if key in res.params.index:
            display_varnames.append(disp)
            param_lookup[disp] = key
    for var, disp in zip(controls, control_display):
        key = f"Q('{var}')"
        if key in res.params.index:
            display_varnames.append(disp)
            param_lookup[disp] = key
    result = _results_to_dict(res, display_varnames=display_varnames, param_lookup=param_lookup)
    return result, formula


def build_regression_latex(
    results: Dict[Tuple[str, str], Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
    *,
    rename_map: Optional[Dict[str, str]] = None,
) -> str:
    """Create a LaTeX regression table from ``regression_plot`` results.

    ``build_regression_latex`` is designed to work out-of-the-box using the
    metadata produced by :func:`regression_plot`.  Passing ``options=None`` or
    ``{}`` will emit a sensible default table that lists every model (simple and
    with controls) that was estimated.  Advanced layouts can still be achieved
    by providing a configuration dictionary.  The most common keys are:

    ``columns``
        A list describing which models should appear as columns.  Each entry is
        a mapping with ``key`` (tuple of ``(y, x)``) and ``model`` (``"simple"``
        or ``"with_controls"``).  ``label`` and ``dependent_label`` override the
        column heading and dependent variable label respectively.
    ``row_order``
        Ordered list of variable display names to keep.  By default all
        available coefficients are displayed.
    ``include_intercept``
        Whether the intercept should be shown (defaults to ``False``).
    ``float_format``
        Format string used for coefficients and summary statistics.
    ``caption`` / ``label``
        Metadata for the LaTeX table environment.
    ``save_path``
        Path on disk to write the LaTeX string to (optional).

    Examples
    --------
    >>> summary = regression_plot(  # doctest: +SKIP
    ...     df,
    ...     x="treatment",
    ...     y="outcome",
    ...     controls=["age", "income"],
    ...     latex_options=True,
    ... )
    >>> print(summary["latex_table"])  # doctest: +SKIP

    Passing ``latex_options=True`` during :func:`regression_plot` automatically
    adds a ``"latex_table"`` entry to the returned dictionary.  To customise the
    layout, supply a dictionary such as ``{"caption": "My Results"}`` or a
    detailed ``{"columns": [...], "row_order": [...]}`` specification.  The
    ``options`` argument here mirrors that structure so you can also refine the
    table post-hoc.

    Parameters
    ----------
    results : dict
        Output of :func:`regression_plot`.
    options : dict, optional
        Table configuration overriding the defaults listed above.
    rename_map : dict, optional
        Mapping from original variable names to pretty labels.
    """

    rename_map = rename_map or {}
    if options is None:
        options = {}
    elif not isinstance(options, dict):
        raise TypeError("options must be a mapping or None")
    else:
        options = dict(options)
    if "caption" not in options:
        options["caption"] = "Regression results"
    if "label" not in options:
        options["label"] = "tab:regression_results"
    columns_spec = options.get("columns")
    if not columns_spec:
        columns_spec = []
        for (y_var, x_var), model_dict in results.items():
            dep_label = rename_map.get(y_var, y_var)
            indep_label = rename_map.get(x_var, x_var)
            if model_dict.get("simple") is not None:
                columns_spec.append({
                    "key": (y_var, x_var),
                    "model": "simple",
                    "label": f"{dep_label} ~ {indep_label}",
                    "dependent_label": dep_label,
                })
            if model_dict.get("with_controls") is not None:
                columns_spec.append({
                    "key": (y_var, x_var),
                    "model": "with_controls",
                    "label": f"{dep_label} ~ {indep_label} + controls",
                    "dependent_label": dep_label,
                })
    models: List[Dict[str, Any]] = []
    for spec in columns_spec:
        key = tuple(spec.get("key", ()))
        if len(key) != 2:
            raise ValueError("Each column specification must include a (y, x) key tuple.")
        if key not in results:
            raise KeyError(f"Result for key {key} not found.")
        model_name = spec.get("model", "simple")
        model_entry = results[key].get(model_name)
        if model_entry is None:
            continue
        label = spec.get("label") or f"Model: {key[0]} ~ {key[1]}"
        dependent_label = spec.get("dependent_label", rename_map.get(key[0], key[0]))
        models.append({
            "label": label,
            "dependent": dependent_label,
            "result": model_entry,
        })
    if not models:
        raise ValueError("No models found for LaTeX table generation.")
    include_intercept = bool(options.get("include_intercept", False))
    float_fmt = options.get("float_format", "{:.3f}")
    row_order = options.get("row_order")
    if row_order is None:
        row_order = []
        for model in models:
            for name in model["result"].get("display_varnames", []):
                if not include_intercept and name == "Intercept":
                    continue
                if name not in row_order:
                    row_order.append(name)
    caption = options.get("caption", "Regression results")
    label = options.get("label", "tab:regression_results")
    include_stats = options.get("include_stats", True)
    include_adj_r2 = options.get("include_adj_r2", False)
    include_controls_row = options.get("include_controls_row", True)
    include_fe_rows = options.get("include_fe_rows", True)
    include_cluster_row = options.get("include_cluster_row", True)
    show_dependent = options.get("show_dependent", True)
    row_end = " " + "\\\\"
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\begin{tabular}{l" + "c" * len(models) + "}", r"\toprule"]
    header = " & " + " & ".join(model["label"] for model in models) + row_end
    lines.append(header)
    if show_dependent:
        dep_row = ["Dependent variable"] + [model["dependent"] for model in models]
        lines.append(" & ".join(dep_row) + row_end)
        lines.append(r"\midrule")
    for row in row_order:
        if not include_intercept and row == "Intercept":
            continue
        row_entries = [row]
        for model in models:
            res = model["result"]
            coef_series = res["coef"]
            se_series = res["se"]
            pvals = res["p"]
            lookup = res.get("param_lookup", {})
            if not isinstance(coef_series, pd.Series):
                coef_series = pd.Series(coef_series, index=res.get("display_varnames"))
            if not isinstance(se_series, pd.Series):
                se_series = pd.Series(se_series, index=res.get("display_varnames"))
            if not isinstance(pvals, pd.Series):
                pvals = pd.Series(pvals, index=res.get("display_varnames"))
            key = lookup.get(row, row)
            if key in coef_series.index:
                formatted = _format_coefficient(
                    float(coef_series[key]),
                    float(se_series[key]) if key in se_series.index else None,
                    float(pvals[key]) if key in pvals.index else None,
                    float_fmt=float_fmt,
                )
            else:
                formatted = "-"
            row_entries.append(formatted)
        lines.append(" & ".join(row_entries) + row_end)
    if include_stats or include_controls_row or include_fe_rows or include_cluster_row:
        lines.append(r"\midrule")
    if include_stats:
        def _fmt_stat(val: Any) -> str:
            return float_fmt.format(val) if pd.notnull(val) else "-"
        obs_row = ["Observations"]
        r2_row = ["R-squared"]
        adj_row = ["Adj. R-squared"]
        for model in models:
            res = model["result"]
            n_val = res.get("n", np.nan)
            obs_row.append(str(int(n_val)) if pd.notnull(n_val) else "-")
            r2_row.append(_fmt_stat(res.get("r2", np.nan)))
            if include_adj_r2:
                adj_row.append(_fmt_stat(res.get("adj_r2", np.nan)))
        lines.append(" & ".join(obs_row) + row_end)
        lines.append(" & ".join(r2_row) + row_end)
        if include_adj_r2:
            lines.append(" & ".join(adj_row) + row_end)
    if include_controls_row:
        ctrl_row = ["Controls"]
        for model in models:
            meta = model["result"].get("metadata", {})
            has_controls = bool(meta.get("controls_included"))
            ctrl_row.append(r"\checkmark" if has_controls else "-")
        lines.append(" & ".join(ctrl_row) + row_end)
    if include_fe_rows:
        entity_row = ["Entity FE"]
        time_row = ["Time FE"]
        for model in models:
            meta = model["result"].get("metadata", {})
            fe = meta.get("fixed_effects", {})
            entity_row.append(r"\checkmark" if fe.get("entity") else "-")
            time_row.append(r"\checkmark" if fe.get("time") else "-")
        lines.append(" & ".join(entity_row) + row_end)
        lines.append(" & ".join(time_row) + row_end)
    if include_cluster_row:
        cluster_row = ["Clustered SE"]
        for model in models:
            meta = model["result"].get("metadata", {})
            fe = meta.get("fixed_effects", {})
            cluster_row.append(r"\checkmark" if fe.get("cluster") else "-")
        lines.append(" & ".join(cluster_row) + row_end)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    latex = "\n".join(lines)
    save_path = options.get("save_path")
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            fh.write(latex)
    return latex

# Set monospace font for consistency
plt.rcParams["font.family"] = "monospace"


def _z(s: pd.Series) -> pd.Series:
    """Return a z‑scored version of a pandas Series (population std)."""
    return (s - s.mean()) / s.std(ddof=0)


def fit_ols(
    y: np.ndarray,
    X: np.ndarray,
    *,
    robust: bool = True,
    varnames: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit an OLS regression using statsmodels and return a dictionary of results.

    When ``varnames`` is supplied and its length matches the number of columns
    in ``X``, the design matrix is converted to a pandas DataFrame with those
    column names so that parameter estimates in the statsmodels summary carry
    meaningful names instead of ``x1``, ``x2``.  Robust HC3 standard errors
    are applied by default.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable values.
    X : ndarray of shape (n, k+1)
        Design matrix including an intercept column.  If ``varnames`` is
        provided, it should contain ``k+1`` names corresponding to the
        columns of ``X``.
    robust : bool, default True
        Use HC3 robust covariance estimates.  If False, classical OLS
        standard errors are used.
    varnames : list of str, optional
        Column names for ``X``.  If provided and valid, these names are used
        in the statsmodels regression and stored in the returned dictionary.

    Returns
    -------
    dict
        Contains coefficient arrays, standard errors, t‑values, p‑values, R²,
        adjusted R², residuals, the fitted statsmodels results object, and
        the variable names used.  If ``varnames`` is ``None`` or mismatched,
        parameter names default to ``const``, ``x1``, ``x2``, etc.
    """
    # Wrap exogenous matrix in DataFrame when names are provided
    if varnames is not None and len(varnames) == X.shape[1]:
        exog = pd.DataFrame(X, columns=varnames)
    else:
        exog = X
        varnames = None
    n, k_plus1 = X.shape
    k = k_plus1 - 1
    model = sm.OLS(y, exog)
    res = model.fit()
    # Apply robust covariance if requested
    if robust:
        try:
            use = res.get_robustcov_results(cov_type="HC3")
        except Exception:
            use = res.get_robustcov_results(cov_type="HC1")
    else:
        use = res
    # Ensure statsmodels returns Series even when given raw ndarrays
    params = use.params
    bse = use.bse
    tvalues = use.tvalues
    pvalues = use.pvalues
    if isinstance(params, np.ndarray):
        if varnames is None:
            try:
                varnames = list(res.model.exog_names)
            except Exception:  # pragma: no cover - very unlikely branch
                varnames = [f"x{i}" for i in range(params.shape[0])]
        params = pd.Series(params, index=varnames)
        bse = pd.Series(np.asarray(bse), index=varnames)
        tvalues = pd.Series(np.asarray(tvalues), index=varnames)
        pvalues = pd.Series(np.asarray(pvalues), index=varnames)
    # Extract statistics
    adj_r2 = res.rsquared_adj
    resid = res.resid
    df_resid = n - k_plus1
    rse = np.sqrt((resid @ resid) / df_resid) if df_resid > 0 else np.nan
    F_stat = res.fvalue if k > 0 else np.nan
    display_names = varnames or list(params.index)
    return {
        "coef": params,
        "se": bse,
        "t": tvalues,
        "p": pvalues,
        "r2": res.rsquared,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "rse": rse,
        "F": F_stat,
        "resid": resid,
        "varnames": varnames,
        "display_varnames": display_names,
        "param_lookup": {name: name for name in display_names},
        "sm_results": res,
    }


def _print_table(res: Dict[str, Any], *, tablefmt: str = "github") -> None:
    """Print a statsmodels summary and a compact coefficient table.

    If the ``tabulate`` library is available, it is used for formatting; otherwise
    pandas' string representation is used.  ``varnames`` should match the
    ordering of the coefficient vector.
    """
    # Print the full statsmodels summary for context
    print(res["sm_results"].summary())
    display_names = res.get("display_varnames") or list(res["coef"].index)
    lookup = res.get("param_lookup", {name: name for name in display_names})
    rows = []
    for name in display_names:
        param_name = lookup.get(name, name)
        if param_name not in res["coef"].index:
            continue
        rows.append({
            "variable": name,
            "coef": res["coef"][param_name],
            "se(HC3)": res["se"][param_name],
            "t": res["t"][param_name],
            "p": res["p"][param_name],
        })
    tbl = pd.DataFrame(rows).set_index("variable") if rows else pd.DataFrame()
    if tabulate is not None:
        print(tabulate(tbl.round(7), headers="keys", tablefmt=tablefmt, showindex=True))
    else:
        print(tbl.round(7).to_string())
    print(f"\nR² = {res['r2']:.4f}, adj. R² = {res['adj_r2']:.4f}, n = {res['n']}")
    print("-" * 60)


def regression_plot(
    df: pd.DataFrame,
    *,
    x: Union[str, Iterable[str]],
    y: Union[str, Iterable[str]],
    controls: Optional[Union[str, Iterable[str]]] = None,
    rename_map: Optional[Dict[str, str]] = None,
    zscore_x: bool = False,
    zscore_y: bool = False,
    bins: int = 20,
    cmap: str = "rainbow",
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 300,
    wrap_width: int = 60,
    show_plots: bool = True,
    tablefmt: str = "github",
    robust: bool = True,
    print_summary: bool = True,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    excess_year_col: Optional[str] = None,
    excess_window: Optional[int] = None,
    excess_mode: str = "difference",
    excess_columns: Optional[Union[str, Iterable[str]]] = None,
    excess_replace: bool = True,
    excess_prefix: str = "",
    fixed_effects: Optional[Dict[str, Any]] = None,
    latex_options: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Run OLS regressions for each combination of ``y`` and ``x`` variables.

    Parameters accept either a string (single variable) or an iterable of
    strings.  For each pair, two models are estimated: one with just the
    independent variable and one including any specified ``controls``.  When
    ``show_plots`` is True, a binned scatter plot with quantile bins and error
    bars is displayed.  If ``zscore_x`` or ``zscore_y`` is True, the respective
    variables are standardised before analysis (but the original variables
    remain untouched in the output).

    ``excess_year_col`` turns on peer-adjusted ("excess") outcome variables.
    Provide the column that defines peer groups (typically a year column) and a
    positive ``excess_window`` describing how many peers before/after should be
    used when computing the rolling mean.  By default every dependent variable
    in ``y`` is adjusted; override with ``excess_columns`` if you also want the
    adjustment applied to other variables.  ``excess_mode`` switches between the
    default difference-from-mean and a ratio-to-mean calculation, while
    ``excess_replace`` controls whether the adjusted columns are automatically
    used in the regression.  ``excess_prefix`` can be used to disambiguate the
    derived columns when running several specifications in succession.

    ``fixed_effects`` allows inclusion of entity/time fixed effects and
    clustered standard errors via statsmodels' formula API.  ``latex_options``
    controls LaTeX output: pass ``True`` (or ``{}``) for a ready-to-use table,
    or supply a configuration dictionary (see :func:`build_regression_latex`
    for the available keys and an end-to-end example).

    Returns a dictionary keyed by ``(y_var, x_var)`` with entries ``'simple'``,
    ``'with_controls'``, and ``'binned_df'`` along with metadata for downstream
    table construction.  When controls are not provided, ``'with_controls'``
    will be ``None``.

    Examples
    --------
    >>> results = regression_plot(  # doctest: +SKIP
    ...     df,
    ...     x="treatment",
    ...     y=["outcome"],
    ...     controls=["age", "income"],
    ...     excess_year_col="year",
    ...     excess_window=2,
    ...     latex_options=True,
    ... )
    >>> list(results.keys())  # doctest: +SKIP
    [('outcome', 'treatment'), 'latex_table']
    """
    x_vars = _ensure_list(x)
    y_vars = _ensure_list(y)
    control_vars = _ensure_list(controls)
    if not x_vars or not y_vars:
        raise ValueError("At least one x and one y variable must be provided.")
    rename_map = dict(rename_map or {})
    prepared_df = df.copy()
    replacements: Dict[str, str] = {}
    if excess_year_col is not None:
        columns = _ensure_list(excess_columns)
        if not columns:
            columns = list(dict.fromkeys(y_vars))  # preserve order, default to y variables
        if excess_window is None or int(excess_window) <= 0:
            raise ValueError("excess_window must be a positive integer when excess_year_col is provided.")
        mode = (excess_mode or "difference").lower()
        prepared_df, replacements = _apply_year_excess(
            prepared_df,
            year_col=excess_year_col,
            window=int(excess_window),
            columns=columns,
            mode=mode,
            replace=bool(excess_replace),
            prefix=excess_prefix,
        )
        suffix = " (excess)" if mode == "difference" else " (ratio)"
        for original, new in replacements.items():
            if original in rename_map and new not in rename_map:
                rename_map[new] = rename_map[original] + suffix
            elif new not in rename_map:
                rename_map[new] = original + suffix
    x_actual = {var: replacements.get(var, var) for var in x_vars}
    y_actual = {var: replacements.get(var, var) for var in y_vars}
    controls_actual = {var: replacements.get(var, var) for var in control_vars}
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    fe_entity = fixed_effects.get("entity") if fixed_effects else None
    fe_time = fixed_effects.get("time") if fixed_effects else None
    cluster_cols = _ensure_list(fixed_effects.get("cluster")) if fixed_effects else []
    include_intercept = True
    if fixed_effects:
        include_intercept = fixed_effects.get("include_intercept")
        if include_intercept is None:
            include_intercept = not (fe_entity or fe_time)
    use_formula = False
    if fixed_effects:
        use_formula = bool(fixed_effects.get("use_formula", False) or fe_entity or fe_time or cluster_cols)
    # In case clustering is requested without other fixed effects
    if cluster_cols:
        use_formula = True
    for y_var in y_vars:
        y_col = y_actual[y_var]
        for x_var in x_vars:
            x_col = x_actual[x_var]
            # Create a copy for each pair to avoid side effects
            data = prepared_df.copy()
            # Pretty names for axes and tables
            y_disp = rename_map.get(y_var, rename_map.get(y_col, y_var))
            x_disp = rename_map.get(x_var, rename_map.get(x_col, x_var))
            ctrl_disp = [rename_map.get(c, rename_map.get(controls_actual[c], c)) for c in control_vars]
            # Ensure variables are numeric; non-numeric rows dropped
            numeric_needed = [x_col, y_col] + [controls_actual[c] for c in control_vars]
            data[numeric_needed] = data[numeric_needed].apply(pd.to_numeric, errors="coerce")
            drop_subset = list(numeric_needed)
            if fe_entity:
                drop_subset.append(fe_entity)
            if fe_time:
                drop_subset.append(fe_time)
            data = data.dropna(subset=drop_subset)
            # Optionally z‑score independent and dependent variables
            x_use = f"{x_col}_z" if zscore_x else x_col
            y_use = f"{y_col}_z" if zscore_y else y_col
            if zscore_x:
                data[x_use] = _z(data[x_col])
                rename_map.setdefault(x_use, f"{x_disp} (z)")
            if zscore_y:
                data[y_use] = _z(data[y_col])
                rename_map.setdefault(y_use, f"{y_disp} (z)")
            # Binned scatter plot
            data["_bin"] = pd.qcut(data[x_use], q=bins, duplicates="drop")
            grp = data.groupby("_bin", observed=True)
            xm = grp[x_use].mean()
            ym = grp[y_use].mean()
            yerr = grp[y_use].apply(sem)
            if show_plots:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax.errorbar(xm, ym, yerr=yerr, fmt="o", color="black",
                            ecolor="black", capsize=3, markersize=6)
                colours = mpl.cm.get_cmap(cmap)(np.linspace(0, 1, len(xm)))
                ax.scatter(xm, ym, c=colours, s=50, zorder=3)
                title = f"{y_disp} vs. {x_disp}"
                if zscore_x:
                    title += " (z‑x)"
                if zscore_y:
                    title += " (z‑y)"
                ax.set_title(textwrap.fill(title, wrap_width))
                ax.set_xlabel(x_disp + (" (z)" if zscore_x else ""))
                ax.set_ylabel(y_disp + (" (z)" if zscore_y else ""))
                ax.grid(alpha=0.3)
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                plt.show()
            # Prepare design matrices and variable names
            y_arr = data[y_use].values
            # Simple model: intercept and primary x variable
            varnames_simple = ["Intercept", x_disp if not zscore_x else f"{x_disp}_z"]
            ctrl_columns = [controls_actual[c] for c in control_vars]
            ctrl_display = ctrl_disp
            simple_res: Dict[str, Any]
            simple_formula = None
            metadata_base = {
                "y": y_var,
                "y_column": y_use,
                "y_display": y_disp,
                "x": x_var,
                "x_column": x_use,
                "x_display": x_disp,
                "controls": control_vars,
                "control_columns": ctrl_columns,
                "control_display": ctrl_disp,
                "controls_included": [],
                "fixed_effects": {
                    "entity": fe_entity,
                    "time": fe_time,
                    "cluster": cluster_cols,
                    "include_intercept": include_intercept,
                },
                "excess_replacements": replacements,
            }
            if use_formula:
                simple_res, simple_formula = _fit_formula_model(
                    data,
                    y=y_use,
                    main_vars=[x_use],
                    main_display=[x_disp if not zscore_x else f"{x_disp} (z)"],
                    controls=[],
                    control_display=[],
                    robust=robust,
                    entity_fe=fe_entity,
                    time_fe=fe_time,
                    include_intercept=include_intercept,
                    cluster_cols=cluster_cols,
                )
            else:
                X_simple = np.column_stack([np.ones(len(data)), data[x_use].values])
                simple_res = fit_ols(y_arr, X_simple, robust=robust, varnames=varnames_simple)
                simple_res["varnames"] = varnames_simple
            if "varnames" not in simple_res and "display_varnames" in simple_res:
                simple_res["varnames"] = simple_res["display_varnames"]
            simple_res.setdefault("metadata", {}).update(metadata_base | {"model": "simple", "formula": simple_formula})
            if print_summary:
                print(f"\n=== Model: {y_disp} ~ {x_disp} ===")
                _print_table(simple_res, tablefmt=tablefmt)
            # Fit controlled model if controls exist
            ctrl_res = None
            ctrl_formula = None
            if control_vars:
                if use_formula:
                    ctrl_res, ctrl_formula = _fit_formula_model(
                        data,
                        y=y_use,
                        main_vars=[x_use],
                        main_display=[x_disp if not zscore_x else f"{x_disp} (z)"],
                        controls=ctrl_columns,
                        control_display=ctrl_display,
                        robust=robust,
                        entity_fe=fe_entity,
                        time_fe=fe_time,
                        include_intercept=include_intercept,
                        cluster_cols=cluster_cols,
                    )
                else:
                    arrays = [np.ones(len(data)), data[x_use].values]
                    for c in ctrl_columns:
                        arrays.append(data[c].values)
                    X_ctrl = np.column_stack(arrays)
                    varnames_ctrl = ["Intercept", x_disp if not zscore_x else f"{x_disp}_z"] + ctrl_disp
                    ctrl_res = fit_ols(y_arr, X_ctrl, robust=robust, varnames=varnames_ctrl)
                    ctrl_res["varnames"] = varnames_ctrl
                if ctrl_res is not None:
                    if "varnames" not in ctrl_res and "display_varnames" in ctrl_res:
                        ctrl_res["varnames"] = ctrl_res["display_varnames"]
                    ctrl_res.setdefault("metadata", {}).update(
                        metadata_base
                        | {
                            "model": "with_controls",
                            "formula": ctrl_formula,
                            "controls_included": ctrl_columns,
                        }
                    )
                    if print_summary:
                        print(f"\n=== Model: {y_disp} ~ {x_disp} + controls ===")
                        _print_table(ctrl_res, tablefmt=tablefmt)
            # Store results keyed by (original y, original x)
            results[(y_var, x_var)] = {
                "simple": simple_res,
                "with_controls": ctrl_res,
                "binned_df": grp[[x_use, y_use]].mean(),
            }
            results[(y_var, x_var)]["metadata"] = metadata_base
    latex_opts: Optional[Dict[str, Any]]
    if isinstance(latex_options, bool):
        latex_opts = {} if latex_options else None
    elif latex_options is None:
        latex_opts = None
    else:
        latex_opts = dict(latex_options)
    if latex_opts is not None:
        latex = build_regression_latex(results, latex_opts, rename_map=rename_map)
        results["latex_table"] = latex
        if latex_opts.get("print", True):
            print(latex)
    return results


def bar_plot(
    categories: Iterable[str],
    values: Iterable[float],
    *,
    title: str = "Classification of Conversations",
    x_label: str = "Classification",
    y_label: str = "Number of Conversations",
    as_percent: bool = False,
    cmap: str = "Reds",
    gradient_start: float = 0.3,
    gradient_end: float = 1.0,
    background_color: str = "#ffffff",
    font_family: str = "monospace",
    figsize: Tuple[float, float] = (16, 7),
    dpi: int = 300,
    label_font_size: int = 12,
    title_font_size: int = 14,
    wrap_width: int = 16,
    rotate_xlabels: bool = False,
    x_label_font_size: int = 12,
    annotation_font_size: int = 10,
    annotation_fontweight: str = "bold",
    precision: int = 3,
) -> None:
    """Draw a bar chart with optional gradient colouring and formatted annotations.

    The behaviour mirrors the original bar plot utility but with clearer
    parameter names.  If ``as_percent`` is True, values are formatted as
    percentages; otherwise large numbers are abbreviated (e.g. 1.2K, 3.4M).
    """
    cats = ["other" if c.strip().lower() == "none" else c for c in categories]
    def fmt(val: float) -> str:
        if as_percent:
            return f"{val:.{precision}g}%"
        if val >= 1e6:
            return f"{val / 1e6:.{precision}g}M"
        if val >= 1e3:
            return f"{val / 1e3:.{precision}g}K"
        return f"{val:.{precision}g}"
    wrapped = [textwrap.fill(c, width=wrap_width) for c in cats]
    colours = plt.cm.get_cmap(cmap)(np.linspace(gradient_start, gradient_end, len(values)))
    plt.style.use("default")
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    bars = ax.bar(wrapped, values, color=colours, edgecolor="black")
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(
            fmt(val),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=annotation_font_size,
            fontweight=annotation_fontweight,
        )
    ax.set_title(textwrap.fill(title, width=100), fontsize=title_font_size, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=label_font_size, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=label_font_size, fontweight="bold")
    if rotate_xlabels:
        plt.xticks(rotation=45, ha="right")
    else:
        plt.xticks(rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=x_label_font_size)
    plt.tight_layout()
    plt.show()


def box_plot(
    data: Union[pd.DataFrame, Dict[str, Iterable[float]], Iterable[Iterable[float]]],
    *,
    labels: Optional[Iterable[str]] = None,
    title: str = "Distribution by Group",
    x_label: str = "Group",
    y_label: str = "Value",
    cmap: str = "viridis",
    gradient_start: float = 0.25,
    gradient_end: float = 0.9,
    background_color: str = "#ffffff",
    font_family: str = "monospace",
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 300,
    notch: bool = False,
    showfliers: bool = False,
    patch_alpha: float = 0.9,
    line_color: str = "#2f2f2f",
    box_linewidth: float = 1.6,
    median_linewidth: float = 2.4,
    annotate_median: bool = True,
    annotation_font_size: int = 10,
    annotation_fontweight: str = "bold",
    wrap_width: int = 22,
    summary_precision: int = 2,
    print_summary: bool = True,
) -> Dict[str, Any]:
    """Render a high-DPI box plot that matches the house style.

    ``data`` may be a tidy DataFrame (columns interpreted as groups), a mapping
    from labels to iterables, or a sequence of iterables.  When ``labels`` is
    omitted, column names or dictionary keys are used automatically.  Numeric
    data are coerced with ``pd.to_numeric`` and missing values are dropped.

    Returns a dictionary containing the Matplotlib ``figure`` and ``ax`` along
    with a ``summary`` DataFrame of descriptive statistics (count, median,
    quartiles and whiskers).  This mirrors the ergonomics of :func:`regression_plot`
    and friends by providing both a styled visual and machine-friendly output.

    Examples
    --------
    >>> out = box_plot(df[["group_a", "group_b"]], title="Score dispersion")  # doctest: +SKIP
    >>> out["summary"].loc["group_a", "median"]  # doctest: +SKIP
    0.42
    """

    if isinstance(data, pd.DataFrame):
        available = list(data.columns)
        if labels is None:
            labels_list = available
        else:
            labels_list = list(labels)
            missing = [label for label in labels_list if label not in data.columns]
            if missing:
                raise KeyError(f"Columns {missing} not found in provided DataFrame.")
        value_arrays = [pd.to_numeric(data[label], errors="coerce").dropna().to_numpy() for label in labels_list]
    elif isinstance(data, dict):
        if labels is None:
            labels_list = list(data.keys())
        else:
            labels_list = list(labels)
        missing = [label for label in labels_list if label not in data]
        if missing:
            raise KeyError(f"Keys {missing} not found in data mapping.")
        value_arrays = [pd.to_numeric(pd.Series(data[label]), errors="coerce").dropna().to_numpy() for label in labels_list]
    else:
        if labels is None:
            try:
                length = len(data)  # type: ignore[arg-type]
            except TypeError:
                raise TypeError("When supplying a sequence of iterables, please provide labels or ensure it has a length.")
            labels_list = [f"Series {i + 1}" for i in range(length)]
        else:
            labels_list = list(labels)
        value_arrays = []
        for idx, series in enumerate(data):
            arr = pd.to_numeric(pd.Series(series), errors="coerce").dropna().to_numpy()
            value_arrays.append(arr)
        if len(value_arrays) != len(labels_list):
            raise ValueError("Number of provided labels does not match the number of series in `data`.")

    if not value_arrays:
        raise ValueError("No data provided for box_plot.")

    plt.style.use("default")
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    bp = ax.boxplot(
        value_arrays,
        labels=[textwrap.fill(str(label), wrap_width) for label in labels_list],
        patch_artist=True,
        notch=notch,
        showfliers=showfliers,
    )

    cmap_obj = cm.get_cmap(cmap)
    colours = cmap_obj(np.linspace(gradient_start, gradient_end, len(value_arrays)))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(line_color)
        patch.set_alpha(patch_alpha)
        patch.set_linewidth(box_linewidth)
    for element in ("whiskers", "caps"):
        for artist in bp[element]:
            artist.set(color=line_color, linewidth=box_linewidth, alpha=0.9)
    for median in bp["medians"]:
        median.set(color=line_color, linewidth=median_linewidth)

    ax.set_title(textwrap.fill(title, width=80), fontsize=14, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    medians = [np.nanmedian(values) if len(values) else np.nan for values in value_arrays]
    if annotate_median:
        for idx, median in enumerate(medians):
            if np.isnan(median):
                continue
            ax.annotate(
                f"{median:.{summary_precision}f}",
                xy=(idx + 1, median),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=annotation_font_size,
                fontweight=annotation_fontweight,
                color=line_color,
            )

    summary_rows = []
    whisker_pairs = [(bp["whiskers"][i], bp["whiskers"][i + 1]) for i in range(0, len(bp["whiskers"]), 2)]
    for label, values, whiskers in zip(labels_list, value_arrays, whisker_pairs):
        if len(values) == 0:
            summary = {"count": 0, "median": np.nan, "mean": np.nan, "std": np.nan, "q1": np.nan, "q3": np.nan, "whisker_low": np.nan, "whisker_high": np.nan}
        else:
            summary = {
                "count": len(values),
                "median": float(np.nanmedian(values)),
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "q1": float(np.nanpercentile(values, 25)),
                "q3": float(np.nanpercentile(values, 75)),
                "whisker_low": float(np.min(whiskers[0].get_ydata())),
                "whisker_high": float(np.max(whiskers[1].get_ydata())),
            }
        summary_rows.append(summary)
    summary_df = pd.DataFrame(summary_rows, index=labels_list)
    if print_summary:
        display_df = summary_df.round(summary_precision)
        if tabulate is not None:
            print(tabulate(display_df, headers="keys", tablefmt="github", showindex=True))
        else:
            print(display_df.to_string())

    plt.tight_layout()
    plt.show()

    return {"figure": fig, "ax": ax, "summary": summary_df}
import os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm

def line_plot(
    df,
    x,                                  # x-axis column (year, date, etc.)
    y=None,                             # numeric column (long); if None with `by`, counts per (x, by)
    by=None,                            # LONG format: category column (mutually exclusive with `series`)
    series=None,                        # WIDE format: list/str of numeric columns to plot
    include=None, exclude=None,         # LONG: filter groups by values in `by`
    top_k=None,                         # keep top-k series by overall plotted weight
    mode='value',                       # 'value' or 'proportion'
    agg='mean',                         # aggregator for duplicates at (x, by): 'mean','median','std','var','cv','se','sum','count'
    smoothing_window=None,              # int (rolling mean window, centered)
    smoothing_method='rolling',         # 'rolling' or 'spline'
    spline_k=3,                         # spline degree (if using 'spline')
    interpolation_points=None,          # optional: upsample to N points across x-range
    # --- presentation ---
    title=None,
    xlabel=None, ylabel=None,
    x_range=None, y_range=None,         # soft clamps for view; aliases below
    xlim=None, ylim=None,
    dpi=400,
    font_family='monospace',
    wrap_width=96,
    grid=True,
    linewidth=2.0,
    cmap_names=None,                    # list of colormap names for distinct series (when no color_map)
    gradient_mode='value',              # 'value' or 'linear' (used only if gradient=True)
    gradient_start=0.35, gradient_end=0.75,
    gradient=True,                      # if False, draw solid lines (respecting color_map/colors)
    color_map=None,                     # dict: {series_name: color_hex}; overrides colormaps
    legend_order=None,                  # list to control legend order
    legend_loc='best',
    alpha=1.0,
    max_lines_per_plot=8,               # batch panels if many series
    save_path=None,                     # file or dir; batches get suffix _setN
    show=True,
):
    """
    Multi-line plotter for *long* or *wide* data with optional proportions, aggregation, smoothing, and batching.

    Quick recipes
    -------------
    LONG (group column):
        line_plot(df, x='year', y='score', by='party', agg='mean')
        # share within each year:
        line_plot(df, x='year', by='party', mode='proportion')  # y=None => counts

    WIDE (several numeric columns already):
        line_plot(df, x='year', series=['dem_score','gop_score'])

    Key behaviors
    -------------
    • mode='value'     : plot values (after aggregating duplicates if long).
    • mode='proportion': within each x, divide each series' value by total across series at that x.
                         (Works for long and wide.)
    • Aggregation (long): when multiple rows share (x, by), combine with `agg`.
    • Smoothing         : centered rolling mean (or B-spline if SciPy available).
    • Colors            : deterministic; prefer `color_map={'A':'#...', 'B':'#...'}` to pin exact hues.
                          If not provided, falls back to colormaps in `cmap_names`.
    • Batching          : if many series, panels are split into sets of `max_lines_per_plot`.

    Parameters worth remembering
    ----------------------------
    - `legend_order=['Democrat','Republican']` for stable legend order.
    - `gradient=False` for solid lines; `True` for aesthetic gradient lines.
    - `top_k=...` to focus on the most important series by overall plotted mass.
    """

    # ---- optional SciPy spline
    try:
        from scipy.interpolate import make_interp_spline
        _spline_available = True
    except Exception:
        _spline_available = False
        if smoothing_method == 'spline':
            print("SciPy not available; using rolling smoothing instead.")
            smoothing_method = 'rolling'

    # Matplotlib basics
    plt.rcParams.update({'font.family': font_family})
    if cmap_names is None:
        cmap_names = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]

    # Exactly one of by / series
    if (by is None) == (series is None):
        raise ValueError("Specify exactly one of `by` (long) OR `series` (wide).")

    work = df.copy()

    # ---- standardize x to something plottable
    if pd.api.types.is_datetime64_any_dtype(work[x]):
        pass
    else:
        # try to coerce numeric; if that fails, leave as-is
        try:
            work[x] = pd.to_numeric(work[x], errors='ignore')
        except Exception:
            pass

    # ---- represent everything as long: (x, _series, _value)
    if series is not None:
        if isinstance(series, (str, int)):
            series = [series]
        missing = [c for c in series if c not in work.columns]
        if missing:
            raise KeyError(f"Missing `series` columns: {missing}")
        long_all = work[[x] + series].melt(id_vars=[x], var_name="_series", value_name="_value")

    else:
        if by not in work.columns:
            raise KeyError(f"`by` column '{by}' not found.")

        if include is not None:
            work = work[work[by].isin(include)]
        if exclude is not None:
            work = work[~work[by].isin(exclude)]

        agg_fns = {
            'mean':  np.mean,
            'median': np.median,
            'std':   np.std,
            'var':   np.var,
            'cv':    lambda s: np.std(s) / (np.mean(s) if np.mean(s)!=0 else np.nan),
            'se':    lambda s: np.std(s) / np.sqrt(max(len(s),1)),
            'sum':   np.sum,
            'count': lambda s: len(s),
        }
        if agg not in agg_fns:
            raise ValueError(f"Unsupported agg '{agg}'.")

        if y is None:
            long_all = (work.groupby([x, by], dropna=False)
                            .size().rename("_value").reset_index())
        else:
            if y not in work.columns:
                raise KeyError(f"`y` column '{y}' not found.")
            tmp = work.rename(columns={y: '_value'})
            long_all = (tmp.groupby([x, by], dropna=False)['_value']
                           .apply(lambda s: agg_fns[agg](s.values)).reset_index())
        long_all = long_all.rename(columns={by: "_series"})

    # ---- compute plotted value
    if mode not in ('value', 'proportion'):
        raise ValueError("mode must be 'value' or 'proportion'.")

    if mode == 'proportion':
        denom = long_all.groupby(x)["_value"].transform(lambda s: s.replace(0, np.nan).sum())
        long_all["_plotval"] = long_all["_value"] / denom
    else:
        long_all["_plotval"] = long_all["_value"]

    # ---- select top_k series (by total plotted value)
    if top_k is not None:
        keep = (long_all.groupby("_series")["_plotval"]
                        .sum(numeric_only=True)
                        .sort_values(ascending=False)
                        .head(int(top_k)).index)
        long_all = long_all[long_all["_series"].isin(set(keep))].copy()

    # ---- sort for plotting
    long_all = long_all.sort_values([x, "_series"])

    # ---- order series
    series_order = legend_order if legend_order else (
        long_all.groupby("_series")["_plotval"].mean(numeric_only=True).sort_values(ascending=False).index.tolist()
    )

    # ---- batching
    if max_lines_per_plot is None or max_lines_per_plot <= 0:
        batches = [series_order]
    else:
        step = int(max_lines_per_plot)
        batches = [series_order[i:i+step] for i in range(0, len(series_order), step)]

    # ---- color resolver
    def _series_color(s, idx):
        if color_map and s in color_map:
            return color_map[s]
        # fall back to palette families by index
        cmap = cm.get_cmap(cmap_names[idx % len(cmap_names)])
        return cmap(0.6)

    figs_axes = []

    def _plot_one(batch_series, batch_idx):
        fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=dpi)
        global_min, global_max = float('inf'), float('-inf')

        for idx, s in enumerate(batch_series):
            sdf = long_all[long_all["_series"] == s].sort_values(x)
            xs = sdf[x].to_numpy()
            ys = sdf["_plotval"].to_numpy()

            # smoothing
            x_s, y_s = xs, ys
            if smoothing_window and smoothing_window > 1 and len(xs) > 1:
                if smoothing_method == 'rolling':
                    y_s = (pd.Series(ys)
                           .rolling(window=int(smoothing_window), min_periods=1, center=True)
                           .mean().to_numpy())
                elif smoothing_method == 'spline' and _spline_available and len(xs) > 2:
                    order = np.argsort(xs)
                    xs_o, ys_o = xs[order], ys[order]
                    k = max(1, min(int(spline_k), len(xs_o) - 1))
                    x_s = np.linspace(xs_o.min(), xs_o.max(),
                                      max(len(xs_o), interpolation_points or len(xs_o)))
                    y_s = make_interp_spline(xs_o, ys_o, k=k)(x_s)

            if interpolation_points and (len(x_s) < interpolation_points):
                xi = np.linspace(np.min(x_s), np.max(x_s), int(interpolation_points))
                yi = np.interp(xi, x_s, y_s)
                x_s, y_s = xi, yi

            if len(y_s) and np.isfinite(y_s).any():
                global_min = min(global_min, np.nanmin(y_s))
                global_max = max(global_max, np.nanmax(y_s))

            color = _series_color(s, idx)

            if gradient:
                # gradient along the line
                pts = np.array([x_s, y_s]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                if isinstance(color, str):
                    # solid color requested but gradient=True → use a subtle light→full alpha ramp
                    from matplotlib.colors import to_rgba
                    base = np.array(to_rgba(color))
                    alphas = np.linspace(gradient_start, gradient_end, max(len(segs), 2))
                    cols = np.tile(base, (len(segs), 1))
                    cols[:, -1] = alphas
                else:
                    # use colormap-driven gradient
                    cmap = cm.get_cmap(cmap_names[idx % len(cmap_names)])
                    if gradient_mode == 'value' and len(y_s) > 1:
                        ymin, ymax = np.nanmin(y_s), np.nanmax(y_s)
                        denom = max((ymax - ymin), 1e-12)
                        norm = (y_s - ymin) / denom
                        seg_vals = (norm[:-1] + norm[1:]) / 2
                        seg_vals = gradient_start + seg_vals * (gradient_end - gradient_start)
                        cols = cmap(seg_vals)
                    else:
                        cols = cmap(np.linspace(gradient_start, gradient_end, max(len(segs), 2)))
                lc = LineCollection(segs, colors=cols, linewidth=linewidth, alpha=alpha, label=str(s))
                ax.add_collection(lc)
            else:
                ax.plot(x_s, y_s, linewidth=linewidth, alpha=alpha, label=str(s), color=color)

        # axis limits (data-driven, then user overrides)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            global_min, global_max = 0.0, 1.0
        if global_max == global_min:
            pad = 1.0 if global_max == 0 else 0.05 * abs(global_max)
            global_min, global_max = global_min - pad, global_max + pad

        xr = xlim if xlim is not None else x_range
        yr = ylim if ylim is not None else y_range
        if xr is None:
            xr = (pd.Series(long_all[x]).min(), pd.Series(long_all[x]).max())
        if yr is None:
            span = (global_max - global_min)
            yr = (global_min - 0.05*span, global_max + 0.05*span)

        ax.set_xlim(xr[0], xr[1])
        ax.set_ylim(yr[0], yr[1])

        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            for sp in ['top','right']:
                ax.spines[sp].set_visible(False)

        # labels & legend
        ttl = title or "line plot"
        if len(batches) > 1:
            ttl = f"{ttl} (set {batch_idx+1}/{len(batches)})"
        ax.set_title(textwrap.fill(ttl, width=wrap_width))
        ax.set_xlabel(xlabel if xlabel is not None else str(x))
        default_ylabel = "share" if mode == 'proportion' else (agg if by is not None else "value")
        ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)

        # legend in requested order
        handles, labels = ax.get_legend_handles_labels()
        if legend_order:
            order_index = {lbl:i for i,lbl in enumerate(labels)}
            order = [order_index[lbl] for lbl in legend_order if lbl in order_index]
            handles = [handles[i] for i in order] + [h for j,h in enumerate(handles) if j not in order]
            labels  = [labels[i] for i in order]  + [l for j,l in enumerate(labels) if j not in order]
        ax.legend(handles, labels, loc=legend_loc, ncol=1, frameon=True)

        plt.tight_layout()

        # save
        if save_path:
            if os.path.isdir(save_path):
                base = (title or "line_plot").strip().replace(" ", "_")
                out = os.path.join(save_path, f"{base}_set{batch_idx+1}.png")
            else:
                root, ext = os.path.splitext(save_path)
                out = f"{root}_set{batch_idx+1}{ext or '.png'}"
            plt.savefig(out, dpi=dpi)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    figs_axes = [ _plot_one(batch, i) for i, batch in enumerate(batches) ]
    return figs_axes
