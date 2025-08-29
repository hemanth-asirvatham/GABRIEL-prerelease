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
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
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
import os, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm

def line_plot(
    df,
    x,                                  # time/x column
    y=None,                              # numeric col (long); if None with `by`, counts per (x, by)
    by=None,                             # category col (long)  XOR with `series`
    series=None,                         # list of columns (wide)
    include=None, exclude=None,          # filter groups (long)
    top_k=None,                          # keep top-k groups by overall weight
    mode='value',                        # 'value' (raw/agg) or 'proportion' (share within each x)
    agg='mean',                          # 'mean','median','std','var','cv','se','sum','count'
    smoothing_window=None,               # int (rolling mean)
    smoothing_method='rolling',          # 'rolling' or 'spline'
    spline_k=3,
    interpolation_points=None,
    # --- presentation ---
    title=None,
    xlabel=None,                         # simple axis label override
    ylabel=None,                         # simple axis label override
    x_range=None, y_range=None,          # (min,max) soft clamps for view
    xlim=None, ylim=None,                # aliases for x_range/y_range
    dpi=400,
    font_family='monospace',
    wrap_width=100,
    grid=True,
    linewidth=2,
    cmap_names=None,                     # list of colormap names
    gradient_mode='value',               # 'value' or 'linear'
    gradient_start=0.4, gradient_end=1.0,
    max_lines_per_plot=5,                # batch panels if too many series
    save_path=None,                      # file or dir; batches get suffix _setN
    show=True,
):
    """
    Multi-line plots from wide or long data.

    Two modes
    ---------
    - mode='value': plot values (or aggregated values) per x.
    - mode='proportion': for long data, plot share of each category within x (value / sum over groups at same x).
                         For wide data, shares are computed across the selected series at each x.

    Formats
    -------
    Wide:  series=[colA, colB,...]; x is your x-axis.
    Long:  by='group_col', y optional (None => counts). Use agg to combine duplicates at (x, by).

    Ranges & Labels
    ---------------
    - x_range=(xmin, xmax) and y_range=(ymin, ymax) limit what’s shown.
      (aliases: xlim, ylim)
    - xlabel / ylabel override axis labels with simple strings.
    """
    # SciPy spline (optional)
    try:
        from scipy.interpolate import make_interp_spline
        _spline_available = True
    except Exception:
        _spline_available = False
        if smoothing_method == 'spline':
            print("SciPy not available; falling back to rolling smoothing.")
            smoothing_method = 'rolling'

    plt.rcParams.update({'font.family': font_family})
    if cmap_names is None:
        cmap_names = ["Reds", "Blues", "Greens", "Purples", "Oranges", "Greys"]

    if (by is None) == (series is None):
        raise ValueError("Specify exactly one of `by` (long) or `series` (wide).")

    work = df.copy()

    # ---- unify to long: (x, _series, _value) ----
    if series is not None:
        if isinstance(series, (str, int)):
            series = [series]
        missing = [c for c in series if c not in work.columns]
        if missing:
            raise KeyError(f"Missing series columns: {missing}")
        long_all = work[[x] + series].melt(id_vars=[x], var_name="_series", value_name="_value")
    else:
        if by not in work.columns:
            raise KeyError(f"`by` column '{by}' not found.")
        if include is not None:
            work = work[work[by].isin(include)]
        if exclude is not None:
            work = work[~work[by].isin(exclude)]

        # choose aggregator
        agg_fns = {
            'mean':  np.mean,
            'median': np.median,
            'std':   np.std,
            'var':   np.var,
            'cv':    lambda s: np.std(s) / (np.mean(s).item() if np.mean(s)!=0 else np.nan),
            'se':    lambda s: np.std(s) / np.sqrt(len(s)),
            'sum':   np.sum,
            'count': lambda s: len(s),
        }
        if agg not in agg_fns:
            raise ValueError(f"Unsupported agg: {agg}")

        if y is None:
            grouped_all = (work.groupby([x, by], dropna=False)
                               .size().rename("_value").reset_index())
        else:
            if y not in work.columns:
                raise KeyError(f"`y` column '{y}' not found.")
            tmp = work.rename(columns={y: '_value'})
            grouped_all = (tmp.groupby([x, by], dropna=False)['_value']
                              .apply(lambda s: agg_fns[agg](s.values)).reset_index())

        grouped_all = grouped_all.rename(columns={by: "_series"})
        long_all = grouped_all

    # ---- proportions if requested ----
    if mode not in ('value', 'proportion'):
        raise ValueError("mode must be 'value' or 'proportion'.")

    if mode == 'proportion':
        denom = long_all.groupby(x)["_value"].transform(lambda s: s.replace(0, np.nan).sum())
        long_all["_plotval"] = long_all["_value"] / denom
    else:
        long_all["_plotval"] = long_all["_value"]

    # top_k after computing plot values (order by total contribution)
    if top_k is not None:
        order_tot = (long_all.groupby("_series")["_plotval"]
                            .sum(numeric_only=True).sort_values(ascending=False))
        keep = set(order_tot.head(int(top_k)).index)
        long_sel = long_all[long_all["_series"].isin(keep)].copy()
    else:
        long_sel = long_all.copy()

    # sort by average plotted value for batching
    order_scores = (long_sel.groupby("_series")["_plotval"]
                           .mean(numeric_only=True)
                           .sort_values(ascending=False))
    series_list = list(order_scores.index)

    # clean/sort x
    if pd.api.types.is_object_dtype(long_sel[x]):
        try:
            long_sel[x] = pd.to_numeric(long_sel[x], errors='ignore')
        except Exception:
            pass
    long_sel = long_sel.sort_values([x, "_series"])

    # batching
    if max_lines_per_plot is None or max_lines_per_plot <= 0:
        batches = [series_list]
    else:
        step = int(max_lines_per_plot)
        batches = [series_list[i:i+step] for i in range(0, len(series_list), step)]

    figs_axes = []

    # helper
    def _plot_one(batch_series, batch_idx):
        fig, ax = plt.subplots(figsize=(9, 5), dpi=dpi)
        global_min, global_max = float('inf'), float('-inf')

        for idx, s in enumerate(batch_series):
            sdf = long_sel[long_sel["_series"] == s].sort_values(x)
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
                    from scipy.interpolate import make_interp_spline
                    y_s = make_interp_spline(xs_o, ys_o, k=k)(x_s)

            if interpolation_points and (len(x_s) < interpolation_points):
                xi = np.linspace(np.min(x_s), np.max(x_s), int(interpolation_points))
                yi = np.interp(xi, x_s, y_s)
                x_s, y_s = xi, yi

            if len(y_s) and np.isfinite(y_s).any():
                global_min = min(global_min, np.nanmin(y_s))
                global_max = max(global_max, np.nanmax(y_s))

            # gradient line
            pts = np.array([x_s, y_s]).T.reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            cmap = cm.get_cmap(cmap_names[idx % len(cmap_names)])
            if gradient_mode == 'value' and len(y_s) > 1:
                ymin, ymax = np.nanmin(y_s), np.nanmax(y_s)
                denom = max((ymax - ymin), 1e-12)
                norm = (y_s - ymin) / denom
                seg_vals = (norm[:-1] + norm[1:]) / 2
                seg_vals = gradient_start + seg_vals * (gradient_end - gradient_start)
                colors = cmap(seg_vals)
            else:
                colors = cmap(np.linspace(gradient_start, gradient_end, max(len(segs), 2)))
            lc = LineCollection(segs, colors=colors, linewidth=linewidth, label=str(s))
            ax.add_collection(lc)

        # axis limits (data-driven, then user overrides)
        if not np.isfinite(global_min) or not np.isfinite(global_max):
            global_min, global_max = 0.0, 1.0
        if global_max == global_min:
            pad = 1.0 if global_max == 0 else 0.05 * abs(global_max)
            global_min, global_max = global_min - pad, global_max + pad
        ymin = global_min - 0.05 * (global_max - global_min)
        ymax = global_max + 0.05 * (global_max - global_min)

        # apply x/y ranges if provided
        xr = xlim if xlim is not None else x_range
        yr = ylim if ylim is not None else y_range
        if xr is None:
            xr = (pd.Series(long_sel[x]).min(), pd.Series(long_sel[x]).max())
        if yr is None:
            yr = (ymin, ymax)

        ax.set_xlim(xr[0], xr[1])
        ax.set_ylim(yr[0], yr[1])

        if grid:
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            for sp in ['top', 'right']:
                ax.spines[sp].set_visible(False)

        # labels: simple by default
        ttl = title or "line plot"
        if len(batches) > 1:
            ttl = f"{ttl} (set {batch_idx+1}/{len(batches)})"
        ax.set_title(textwrap.fill(ttl, width=wrap_width))
        ax.set_xlabel(xlabel if xlabel is not None else str(x))
        # default ylabel: simple; no mode-jargon
        default_ylabel = "share" if mode == 'proportion' else (agg if by is not None else "value")
        ax.set_ylabel(ylabel if ylabel is not None else default_ylabel)
        ax.legend(loc='best', ncol=1)
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
