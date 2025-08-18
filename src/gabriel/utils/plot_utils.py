"""
Enhanced Gabriel Visualisation Utilities
=======================================

This module refines the original plotting utilities to provide:

* OLS regressions via statsmodels with meaningful coefficient names
  (no more ``x1``, ``x2``) and optional robust standard errors.
* Binned scatter plots that support multiple independent variables via
  ``controls`` and allow custom axis limits.
* Bar and line plots with a variety of customisation options.

The functions mirror the earlier API but with cleaner parameter names
and additional features.  For Python 3.12 and SciPy 1.16+, use
``statsmodels>=0.14.5`` to avoid import errors【684365170165574†L26-L32】.
"""

from __future__ import annotations

import textwrap
from typing import Iterable, Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import sem

try:
    from tabulate import tabulate  # type: ignore
except ModuleNotFoundError:
    tabulate = None  # fallback when tabulate isn't installed

import statsmodels.api as sm

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
    # Extract statistics
    adj_r2 = res.rsquared_adj
    resid = res.resid
    df_resid = n - k_plus1
    rse = np.sqrt((resid @ resid) / df_resid) if df_resid > 0 else np.nan
    F_stat = res.fvalue if k > 0 else np.nan
    return {
        "coef": use.params,
        "se": use.bse,
        "t": use.tvalues,
        "p": use.pvalues,
        "r2": res.rsquared,
        "adj_r2": adj_r2,
        "n": n,
        "k": k,
        "rse": rse,
        "F": F_stat,
        "resid": resid,
        "varnames": varnames,
        "sm_results": res,
    }


def _print_table(res: Dict[str, Any], varnames: List[str], *, tablefmt: str = "github") -> None:
    """Print a statsmodels summary and a compact coefficient table.

    If the ``tabulate`` library is available, it is used for formatting; otherwise
    pandas' string representation is used.  ``varnames`` should match the
    ordering of the coefficient vector.
    """
    # Print the full statsmodels summary for context
    print(res["sm_results"].summary())
    tbl = pd.DataFrame({
        "coef": res["coef"],
        "se(HC3)": res["se"],
        "t": res["t"],
        "p": res["p"],
    }, index=varnames)
    if tabulate is not None:
        print(tabulate(tbl.round(7), headers="keys", tablefmt=tablefmt, showindex=True))
    else:
        print(tbl.round(7).to_string())
    print(f"\nR² = {res['r2']:.4f}, adj. R² = {res['adj_r2']:.4f}, n = {res['n']}")
    print("-" * 60)


def regression_plot(
    df: pd.DataFrame,
    *,
    x_vars: Iterable[str],
    y_vars: Iterable[str],
    controls: Optional[Iterable[str]] = None,
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
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Run OLS regressions for each combination of ``y`` in ``y_vars`` and ``x`` in ``x_vars``.

    For each pair, two models are estimated: one with just the independent
    variable and one including any specified ``controls``.  When
    ``show_plots`` is True, a binned scatter plot with quantile bins and error
    bars is displayed.  If ``zscore_x`` or ``zscore_y`` is True, the
    respective variables are standardised before analysis (but the original
    variables remain untouched in the output).

    Additional optional axis limits can be set via ``xlim`` and ``ylim``.

    Returns a dictionary keyed by ``(y_var, x_var)`` with entries ``'simple'``,
    ``'with_controls'``, and ``'binned_df'``.  The regression results include
    the full statsmodels result, coefficient arrays, and residuals.  When
    controls are not provided, ``'with_controls'`` will be ``None``.
    """
    controls = list(controls) if controls else []
    rename_map = rename_map or {}
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for y in y_vars:
        for x in x_vars:
            # Create a copy for each pair to avoid side effects
            data = df.copy()
            # Pretty names for axes and tables
            y_disp = rename_map.get(y, y)
            x_disp = rename_map.get(x, x)
            ctrl_disp = [rename_map.get(c, c) for c in controls]
            # Ensure variables are numeric; non-numeric rows dropped
            needed = [x, y] + controls
            data[needed] = data[needed].apply(pd.to_numeric, errors="coerce")
            data = data.dropna(subset=needed)
            # Optionally z‑score independent and dependent variables
            x_use = f"{x}_z" if zscore_x else x
            y_use = f"{y}_z" if zscore_y else y
            if zscore_x:
                data[x_use] = _z(data[x])
            if zscore_y:
                data[y_use] = _z(data[y])
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
            X_simple = np.column_stack([np.ones(len(data)), data[x_use].values])
            varnames_simple = ["Intercept", x_disp if not zscore_x else f"{x_disp}_z"]
            # Controlled model: intercept, primary x, then controls
            X_ctrl = None
            varnames_ctrl: List[str] | None = None
            if controls:
                arrays = [np.ones(len(data)), data[x_use].values]
                for c in controls:
                    arrays.append(data[c].values)
                X_ctrl = np.column_stack(arrays)
                varnames_ctrl = ["Intercept", x_disp if not zscore_x else f"{x_disp}_z"] + ctrl_disp
            # Fit simple model
            simple_res = fit_ols(y_arr, X_simple, robust=robust, varnames=varnames_simple)
            simple_res["varnames"] = varnames_simple
            if print_summary:
                print(f"\n=== Model: {y_disp} ~ {x_disp} ===")
                _print_table(simple_res, varnames_simple, tablefmt=tablefmt)
            # Fit controlled model if controls exist
            ctrl_res = None
            if controls and X_ctrl is not None and varnames_ctrl is not None:
                ctrl_res = fit_ols(y_arr, X_ctrl, robust=robust, varnames=varnames_ctrl)
                ctrl_res["varnames"] = varnames_ctrl
                if print_summary:
                    print(f"\n=== Model: {y_disp} ~ {x_disp} + controls ===")
                    _print_table(ctrl_res, varnames_ctrl, tablefmt=tablefmt)
            # Store results keyed by (original y, original x)
            results[(y, x)] = {
                "simple": simple_res,
                "with_controls": ctrl_res,
                "binned_df": grp[[x_use, y_use]].mean(),
            }
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


def line_plot(
    x: Iterable[Any],
    ys: Iterable[Iterable[float]],
    *,
    labels: Optional[Iterable[str]] = None,
    title: str = "Line Plot",
    x_label: str = "X",
    y_label: str = "Y",
    colors: Optional[Iterable[str]] = None,
    background_color: str = "#ffffff",
    font_family: str = "monospace",
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 300,
    label_font_size: int = 12,
    title_font_size: int = 14,
    wrap_width: int = 60,
    legend: bool = True,
) -> None:
    """Plot one or more line series against a shared x‑axis.

    Parameters
    ----------
    x : iterable
        Values on the x‑axis.
    ys : iterable of iterables
        Each element is a sequence of y values corresponding to x.
    labels : iterable of str, optional
        Labels for each line series.  If None, lines are labelled "Series i".
    colors : iterable of str, optional
        Colours for each line.  Defaults to Matplotlib’s color cycle if None.
    legend : bool, default True
        Whether to display a legend.
    """
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(ys))]
    if colors is None:
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        colours = list(colors)
    for i, y_vals in enumerate(ys):
        ax.plot(x, y_vals, label=labels[i], color=colours[i % len(colours)])
    ax.set_title(textwrap.fill(title, wrap_width), fontsize=title_font_size, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=label_font_size, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=label_font_size, fontweight="bold")
    ax.grid(alpha=0.3)
    if legend:
        ax.legend()
    plt.tight_layout()
    plt.show()
