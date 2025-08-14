"""
mapmaker.py
~~~~~~~~~~~~

This module implements a self‑contained helper class for generating
choropleth maps from ranked GABRIEL outputs.  The ``MapMaker`` can
produce county‑, state‑ or country‑level maps depending on the
``map_type`` parameter, and it optionally normalises values as
z‑scores.  All mapping logic resides in this file, replacing the need
for the separate ``create_county_choropleth`` function.
"""

from __future__ import annotations

import os
import json
import requests
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px


class MapMaker:
    """Utility for generating geographic choropleth maps from a data frame.

    The input data frame should contain at least one column of numeric
    scores and one or more columns identifying the geographic unit
    (county FIPS codes, two‑letter US state abbreviations or ISO‑3
    country codes).  Individual maps are rendered using Plotly and
    written to ``save_dir`` with names derived from the value column.

    Parameters
    ----------
    df:
        DataFrame containing the data to plot.  Each row should
        correspond to a geographic region.
    fips_col:
        Name of the column containing five‑digit county FIPS codes.
    state_col:
        Name of the column containing two‑letter US state abbreviations.
    country_col:
        Name of the column containing ISO‑3 country codes.
    save_dir:
        Directory to which map files will be written.  If ``None``,
        a ``maps`` subdirectory in the current working directory is used.
    z_score:
        Whether to convert values to z‑scores before plotting.
    color_scale:
        Name of the Plotly colour scale to apply.  Defaults to
        ``"RdBu"`` (diverging) when z‑scores are enabled and
        ``"Viridis"`` otherwise.
    map_type:
        Determines the map produced: ``"county"``, ``"state"``
        or ``"country"`` (with ``"global"`` as an alias).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fips_col: Optional[str] = None,
        state_col: Optional[str] = None,
        country_col: Optional[str] = None,
        save_dir: Optional[str] = None,
        z_score: bool = True,
        color_scale: str = "RdBu",
        map_type: str = "county",
    ) -> None:
        self.df = df.copy()
        self.fips_col = fips_col
        self.state_col = state_col
        self.country_col = country_col

        # normalise map_type and validate
        map_type = map_type.lower()
        if map_type not in {"county", "state", "country", "global"}:
            raise ValueError(
                "map_type must be one of 'county', 'state', 'country' or 'global'"
            )
        self.map_type = "country" if map_type == "global" else map_type

        # choose save directory
        if save_dir is None:
            save_dir = os.path.join(os.getcwd(), "maps")
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        self.z_score = z_score
        self.color_scale = color_scale

    def _compute_zscore(self, values: np.ndarray) -> np.ndarray:
        """Compute z‑scores with safe handling of NaNs and constant arrays."""
        vals = values.astype(float)
        if len(vals) > 1 and np.nanstd(vals) > 0:
            return (vals - np.nanmean(vals)) / np.nanstd(vals)
        return np.zeros_like(vals)

    def _create_state_choropleth(
        self,
        df: pd.DataFrame,
        state_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a state‑level choropleth."""
        plot_col = value_col
        colour_scale = self.color_scale
        df_local = df.copy()
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"
        fig = px.choropleth(
            df_local,
            locations=state_col,
            locationmode="USA-states",
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="usa",
            hover_data={state_col: True, value_col: True},
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            fig.write_image(save_path, scale=3)
        else:
            fig.write_html(save_path)

    def _create_country_choropleth(
        self,
        df: pd.DataFrame,
        country_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a global choropleth using ISO‑3 codes."""
        plot_col = value_col
        colour_scale = self.color_scale
        df_local = df.copy()
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"
        fig = px.choropleth(
            df_local,
            locations=country_col,
            locationmode="ISO-3",
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="world",
            hover_data={country_col: True, value_col: True},
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            fig.write_image(save_path, scale=3)
        else:
            fig.write_html(save_path)

    def _create_county_choropleth(
        self,
        df: pd.DataFrame,
        fips_col: str,
        value_col: str,
        title: str,
        save_path: str,
    ) -> None:
        """Create and save a county‑level choropleth with FIPS codes.

        This method inlines the logic of the old ``create_county_choropleth``
        function to avoid external dependencies.  It downloads a GeoJSON of
        U.S. counties on first use and caches it in ``~/.cache/county_geo.json``.
        """
        # pad FIPS codes to five digits
        df_local = df.copy()
        df_local[fips_col] = df_local[fips_col].astype(str).str.zfill(5)

        # find a county name column for hover text
        county_col = None
        for cand in ["county", "County", "region", "Region"]:
            if cand in df_local.columns:
                county_col = cand
                break
        if county_col is None:
            county_col = "_county_name"
            df_local[county_col] = ""

        # load or download county GeoJSON
        geojson_url = (
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        )
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "county_geo.json")
        if not os.path.exists(cache_path):
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            resp = requests.get(geojson_url, timeout=30)
            resp.raise_for_status()
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
        with open(cache_path, encoding="utf-8") as f:
            counties = json.load(f)

        # prepare value column (with optional z‑score)
        plot_col = value_col
        colour_scale = self.color_scale
        if self.z_score:
            zs = self._compute_zscore(df_local[value_col].values)
            plot_col = f"_zscore_{value_col}"
            df_local[plot_col] = zs
            colour_scale = "RdBu" if self.color_scale == "RdBu" else "PuOr"

        hover_data = {county_col: True, fips_col: True, value_col: True}
        fig = px.choropleth(
            df_local,
            geojson=counties,
            locations=fips_col,
            color=plot_col,
            color_continuous_scale=colour_scale,
            scope="usa",
            hover_data=hover_data,
        )
        if self.z_score:
            fig.update_coloraxes(cmid=0)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            fig.write_image(save_path, scale=3)
        else:
            fig.write_html(save_path)

    def make_maps(self, value_cols: Iterable[str]) -> None:
        """Generate and save maps for each specified numeric column.

        The map type is determined by ``self.map_type``: ``"county"``
        uses FIPS codes; ``"state"`` uses two‑letter abbreviations; and
        ``"country"`` uses ISO‑3 codes.
        """
        for value_col in value_cols:
            if self.map_type == "county":
                if not self.fips_col:
                    raise ValueError("fips_col must be provided for county maps")
                fname = f"county_map_{value_col}.html"
                save_path = os.path.join(self.save_dir, fname)
                self._create_county_choropleth(
                    self.df,
                    self.fips_col,
                    value_col,
                    title=f"County Map for {value_col}",
                    save_path=save_path,
                )
            elif self.map_type == "state":
                if not self.state_col:
                    raise ValueError("state_col must be provided for state maps")
                fname = f"state_map_{value_col}.html"
                save_path = os.path.join(self.save_dir, fname)
                self._create_state_choropleth(
                    self.df,
                    self.state_col,
                    value_col,
                    title=f"State Map for {value_col}",
                    save_path=save_path,
                )
            elif self.map_type == "country":
                if not self.country_col:
                    raise ValueError("country_col must be provided for country maps")
                fname = f"country_map_{value_col}.html"
                save_path = os.path.join(self.save_dir, fname)
                self._create_country_choropleth(
                    self.df,
                    self.country_col,
                    value_col,
                    title=f"Country Map for {value_col}",
                    save_path=save_path,
                )
            else:
                # should not happen due to validation in __init__
                raise ValueError(f"Unsupported map type: {self.map_type}")


def create_county_choropleth(
    df: pd.DataFrame,
    *,
    fips_col: str,
    value_col: str,
    title: str,
    save_path: str,
    z_score: bool = True,
) -> None:
    """Backward compatible helper to generate a county-level choropleth.

    This thin wrapper instantiates :class:`MapMaker` and delegates to its
    internal implementation.  It mirrors the signature of the legacy
    ``create_county_choropleth`` function used elsewhere in the codebase.
    """

    mm = MapMaker(df, fips_col=fips_col, z_score=z_score, save_dir=None, map_type="county")
    mm._create_county_choropleth(df, fips_col, value_col, title, save_path)
