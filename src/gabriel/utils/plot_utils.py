"""
Gabriel Visualisation Utilities
==============================

This module collects helper functions for exploratory analysis and
presentation.  It offers:

* Robust OLS regressions with binned scatter plots and summary tables
* Customisable bar charts
* Simple line plots

Functions
---------
- ``fit_ols``: Fit an OLS model using ``statsmodels`` with optional robust
  standard errors.
- ``regression_plot``: Run regressions for combinations of dependent and
  independent variables, optionally show binned scatter plots and print
  statsmodels summary tables.
- ``bar_plot``: Draw a bar chart with gradient colours and value
  annotations.
- ``line_plot``: Plot one or more series against a shared x‑axis.

Notes
-----
For compatibility with Python 3.12 and SciPy 1.16+, ensure that you
install ``statsmodels>=0.14.5``【684365170165574†L26-L32】.
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
    tabulate = None  # fallback
import statsmodels.api as sm

# Use monospace font for console output
plt.rcParams["font.family"] = "monospace"

# Helper: z-score a pandas Series.  We define it here rather than depend
# on a closure so that it is available to regression_plot.  Uses population
# standard deviation (ddof=0) by default.
def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


# -----------------------------------------------------------------------------
# Core regression
# -----------------------------------------------------------------------------

def fit_ols(y: np.ndarray, X: np.ndarray, *, robust: bool = True) -> Dict[str, Any]:
    """Fit an OLS regression using statsmodels and return a results dict.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Dependent variable.
    X : ndarray of shape (n, k+1)
        Design matrix with intercept in the first column.
    robust : bool, default True
        Use HC3 robust standard errors.

    Returns
    -------
    dict
        Keys include 'coef', 'se', 't', 'p', 'r2', 'adj_r2', 'n', 'k', 'rse',
        'F', 'resid', and 'sm_results' (the full statsmodels results object).
    """
    n, k_plus1 = X.shape
    k = k_plus1 - 1
    model = sm.OLS(y, X)
    res = model.fit()
    if robust:
        try:
            use = res.get_robustcov_results(cov_type="HC3")
        except Exception:
            use = res.get_robustcov_results(cov_type="HC1")
    else:
        use = res
    adj_r2 = res.rsquared_adj
    resid = res.resid
    df_resid = n - k_plus1
    rse = np.sqrt((resid @ resid) / df_resid) if df_resid > 0 else np.nan
    F_stat = res.fvalue if k > 0 else np.nan
    return {
        'coef': use.params,
        'se': use.bse,
        't': use.tvalues,
        'p': use.pvalues,
        'r2': res.rsquared,
        'adj_r2': adj_r2,
        'n': n,
        'k': k,
        'rse': rse,
        'F': F_stat,
        'resid': resid,
        'varnames': None,
        'sm_results': res
    }


def _print_table(res: Dict[str, Any], varnames: List[str], *, tablefmt: str = "github") -> None:
    """Print a regression result as a table with tabulate and statsmodels summary."""
    # statsmodels summary
    print(res['sm_results'].summary())
    # tabulated coefficients
    tbl = pd.DataFrame({
        'coef': res['coef'],
        'se(HC3)': res['se'],
        't': res['t'],
        'p': res['p']
    }, index=varnames)
    if tabulate is not None:
        print(tabulate(tbl.round(7), headers="keys", tablefmt=tablefmt, showindex=True))
    else:
        # Fallback: use pandas built-in formatting
        print(tbl.round(7).to_string())
    print(f"\nR² = {res['r2']:.4f}, adj. R² = {res['adj_r2']:.4f}, n = {res['n']}")
    print("-" * 60)


# -----------------------------------------------------------------------------
# Regression plotting wrapper
# -----------------------------------------------------------------------------

def regression_plot(df: pd.DataFrame, *, x_vars: Iterable[str], y_vars: Iterable[str],
                    controls: Optional[Iterable[str]] = None,
                    rename_map: Optional[Dict[str, str]] = None,
                    zscore_x: bool = False,
                    zscore_y: bool = False,
                    bins: int = 20,
                    cmap: str = 'rainbow',
                    figsize: Tuple[float,float] = (8,6),
                    dpi: int = 300,
                    wrap_width: int = 60,
                    show_plots: bool = True,
                    tablefmt: str = 'github',
                    robust: bool = True,
                    print_summary: bool = True) -> Dict[Tuple[str,str], Dict[str, Any]]:
    """Run regressions for each combination of y in y_vars and x in x_vars.

    Parameters
    ----------
    df : DataFrame
        DataFrame containing all variables.
    x_vars, y_vars : iterables of str
        Independent and dependent variable names.  All pairwise combinations
        will be run.
    controls : iterable of str, optional
        Control variables included in every specification.
    rename_map : dict, optional
        Mapping from raw column names to pretty names for axes and tables.
    zscore_x, zscore_y : bool, default False
        Whether to z‑score independent or dependent variables.
    bins : int, default 20
        Number of quantile bins for the binned scatter.
    cmap : str, default 'rainbow'
        Colormap for scatter points.
    figsize, dpi : tuple, default (8,6), 300
        Size and resolution of the scatter plots.
    wrap_width : int, default 60
        Maximum line width for wrapped titles.
    show_plots : bool, default True
        If False, skip plotting but still print regression tables.
    tablefmt : str, default 'github'
        Format for tabulated coefficient tables.

    robust : bool, default True
        Whether to use HC3 robust standard errors.  Passed down to
        ``fit_ols``.
    print_summary : bool, default True
        If True, print both the statsmodels summary and a small table for
        each model.  If False, skip printing but still return results.

    Returns
    -------
    dict
        Mapping from (y, x) tuples to dictionaries with keys 'simple' and
        'with_controls' containing regression results, plus 'binned_df'.
    """
    controls = list(controls) if controls else []
    rename_map = rename_map or {}
    results: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for y in y_vars:
        for x in x_vars:
            # Create copy for each pair
            data = df.copy()
            # Resolve display names
            y_disp = rename_map.get(y, y)
            x_disp = rename_map.get(x, x)
            ctrl_disp = [rename_map.get(c, c) for c in controls]
            # ensure numeric
            needed = [x, y] + controls
            data[needed] = data[needed].apply(pd.to_numeric, errors='coerce')
            data = data.dropna(subset=needed)
            # z‑scoring
            x_use = f"{x}_z" if zscore_x else x
            y_use = f"{y}_z" if zscore_y else y
            if zscore_x:
                data[x_use] = _z(data[x])
            if zscore_y:
                data[y_use] = _z(data[y])
            # binned scatter
            data['_bin'] = pd.qcut(data[x_use], q=bins, duplicates='drop')
            grp = data.groupby('_bin', observed=True)
            xm = grp[x_use].mean()
            ym = grp[y_use].mean()
            yerr = grp[y_use].apply(sem)
            if show_plots:
                fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
                ax.errorbar(xm, ym, yerr=yerr, fmt='o', color='black',
                            ecolor='black', capsize=3, markersize=6)
                colours = mpl.cm.get_cmap(cmap)(np.linspace(0,1,len(xm)))
                ax.scatter(xm, ym, c=colours, s=50, zorder=3)
                title = f"{y_disp} vs. {x_disp}"
                if zscore_x: title += " (z‑x)"
                if zscore_y: title += " (z‑y)"
                ax.set_title(textwrap.fill(title, wrap_width))
                ax.set_xlabel(x_disp + (" (z)" if zscore_x else ""))
                ax.set_ylabel(y_disp + (" (z)" if zscore_y else ""))
                ax.grid(alpha=0.3)
                plt.show()
            # design matrices
            y_arr = data[y_use].values
            X_simple = np.column_stack([np.ones(len(data)), data[x_use].values])
            X_ctrl = np.column_stack([np.ones(len(data)), data[x_use].values] + [data[c].values for c in controls])
            # simple model
            simple_res = fit_ols(y_arr, X_simple, robust=robust)
            simple_res['varnames'] = ['Intercept', x_disp if not zscore_x else f"{x_disp}_z"]
            if print_summary:
                print(f"\n=== Model: {y_disp} ~ {x_disp} ===")
                _print_table(simple_res, simple_res['varnames'], tablefmt=tablefmt)
            # controlled model
            ctrl_res = None
            if controls:
                ctrl_res = fit_ols(y_arr, X_ctrl, robust=robust)
                ctrl_res['varnames'] = ['Intercept', x_disp if not zscore_x else f"{x_disp}_z"] + ctrl_disp
                if print_summary:
                    print(f"\n=== Model: {y_disp} ~ {x_disp} + controls ===")
                    _print_table(ctrl_res, ctrl_res['varnames'], tablefmt=tablefmt)
            results[(y, x)] = {
                'simple': simple_res,
                'with_controls': ctrl_res,
                'binned_df': grp[[x_use, y_use]].mean()
            }
    return results


# -----------------------------------------------------------------------------
# Bar plot (adapted from provided code)
# -----------------------------------------------------------------------------

def bar_plot(categories: Iterable[str], values: Iterable[float], *,
             title: str = "Classification of Conversations",
             x_label: str = "Classification",
             y_label: str = "Number of Conversations",
             as_percent: bool = False,
             cmap: str = "Reds",
             gradient_start: float = 0.3,
             gradient_end: float = 1.0,
             background_color: str = "#ffffff",
             font_family: str = "monospace",
             figsize: Tuple[float,float] = (16,7),
             dpi: int = 300,
             label_font_size: int = 12,
             title_font_size: int = 14,
             wrap_width: int = 16,
             rotate_xlabels: bool = False,
             x_label_font_size: int = 12,
             annotation_font_size: int = 10,
             annotation_fontweight: str = "bold",
             precision: int = 3) -> None:
    """Draw a bar chart with optional gradient colours and value annotations."""
    cats = ["other" if c.strip().lower() == "none" else c for c in categories]
    def fmt(val: float) -> str:
        if as_percent:
            return f"{val:.{precision}g}%"
        if val >= 1e6:
            return f"{val/1e6:.{precision}g}M"
        if val >= 1e3:
            return f"{val/1e3:.{precision}g}K"
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
            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
            fontsize=annotation_font_size, fontweight=annotation_fontweight,
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


# -----------------------------------------------------------------------------
# Simple line plot
# -----------------------------------------------------------------------------

def line_plot(x: Iterable[Any], ys: Iterable[Iterable[float]], *,
              labels: Optional[Iterable[str]] = None,
              title: str = "Line Plot",
              x_label: str = "X",
              y_label: str = "Y",
              colors: Optional[Iterable[str]] = None,
              background_color: str = "#ffffff",
              font_family: str = "monospace",
              figsize: Tuple[float,float] = (10,6),
              dpi: int = 300,
              label_font_size: int = 12,
              title_font_size: int = 14,
              wrap_width: int = 60,
              legend: bool = True) -> None:
    """Plot one or more line series against a common x‑axis.

    Parameters
    ----------
    x : iterable
        Values on the x‑axis.
    ys : iterable of iterables
        Each element is a sequence of y values corresponding to x.
    labels : iterable of str, optional
        Labels for each line series.  If None, lines are labelled "Series i".
    title : str
        Plot title.
    x_label, y_label : str
        Axis labels.
    colors : iterable of str, optional
        Colours for each line.  Defaults to Matplotlib cycle.
    background_color : str
        Figure and axes background colour.
    font_family : str
        Global font family.
    figsize, dpi : tuple
        Figure size and resolution.
    label_font_size : int
        Axis label font size.
    title_font_size : int
        Title font size.
    wrap_width : int
        Line wrap width for the title.
    legend : bool, default True
        Show legend if True.
    """
    plt.rcParams['font.family'] = font_family
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)
    if labels is None:
        labels = [f"Series {i+1}" for i in range(len(ys))]
    if colors is None:
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    else:
        colours = colors
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
