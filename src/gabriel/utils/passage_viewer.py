import ast
import colorsys
import math
import html
import json
import random
import re
import uuid
from string import Template
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Literal,
    Set,
)

import pandas as pd
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


def _generate_distinct_colors(n: int) -> List[str]:
    """Generate ``n`` visually distinct hex colors.

    This helper is shared by both the rich ``tkinter`` viewer and the simpler
    HTML based viewer used in headless environments such as Google Colab.
    """

    base_colors: List[str] = []
    if plt is not None:
        if n <= 20:
            cmap = plt.get_cmap("tab20")
            for i in range(n):
                rgb = cmap(i)[:3]
                base_colors.append(
                    "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                )
            return base_colors
        else:
            cmap = plt.get_cmap("tab20")
            for i in range(20):
                rgb = cmap(i)[:3]
                base_colors.append(
                    "#{:02x}{:02x}{:02x}".format(
                        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                    )
                )

    for i in range(len(base_colors), n):
        hue = (i * 1.0 / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        base_colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )
    return base_colors[:n]


@dataclass(frozen=True)
class _AttributeRequest:
    column: str
    label: str
    dynamic: bool = False
    description: Optional[str] = None


@dataclass(frozen=True)
class _AttributeSpec:
    column: str
    label: str
    kind: Literal["snippet", "boolean", "numeric", "text"]
    dynamic: bool = False
    description: Optional[str] = None

def _coerce_category_spec(
    categories: Optional[Union[Sequence[Any], Any]]
) -> Optional[Union[List[str], str]]:
    """Normalize the ``categories`` argument into a predictable form."""

    if categories is None:
        return None

    if isinstance(categories, str):
        return categories if categories == "coded_passages" else [categories]

    if isinstance(categories, Iterable):
        normalized: List[str] = []
        for item in categories:
            if item is None:
                continue
            normalized.append(str(item))
        return normalized

    return [str(categories)]


def _normalize_attribute_requests(
    attributes: Optional[Union[Mapping[str, Any], Sequence[Any], Any]]
) -> List[_AttributeRequest]:
    """Coerce attribute selections into a structured list."""

    if attributes is None:
        return []

    label_overrides: Dict[str, str] = {}
    descriptions: Dict[str, str] = {}
    if isinstance(attributes, Mapping):
        iterable: Iterable[Any] = attributes.keys()
        for key, value in attributes.items():
            if value is None:
                continue
            column_key = str(key)
            if isinstance(value, Mapping):
                label_hint = str(value.get("label", "")).strip()
                desc_hint = str(value.get("description", "")).strip()
                if label_hint:
                    label_overrides[column_key] = label_hint
                if desc_hint:
                    descriptions[column_key] = desc_hint
                continue
            text = str(value).strip()
            if not text:
                continue
            descriptions[column_key] = text
    elif isinstance(attributes, (str, bytes)):
        iterable = [attributes]
    elif isinstance(attributes, Iterable):
        iterable = attributes
    else:
        iterable = [attributes]

    requests: List[_AttributeRequest] = []
    seen: set[Tuple[str, bool]] = set()
    for entry in iterable:
        dynamic = False
        label_hint: Optional[str] = None
        description_hint: Optional[str] = None
        if isinstance(entry, tuple) and entry:
            column = str(entry[0])
            label_hint = str(entry[1]) if len(entry) > 1 else None
        elif isinstance(entry, list) and entry:
            column = str(entry[0])
            label_hint = str(entry[1]) if len(entry) > 1 else None
        elif isinstance(entry, tuple) and not entry:
            continue
        elif isinstance(entry, Mapping):
            for key, value in entry.items():
                column = str(key)
                override = str(value).strip() if value is not None else ""
                dynamic = column == "coded_passages"
                pretty = override or column.replace("_", " ").title()
                identity = (column, dynamic)
                if identity in seen:
                    continue
                seen.add(identity)
                requests.append(
                    _AttributeRequest(
                        column=column,
                        label=pretty,
                        dynamic=dynamic,
                        description=descriptions.get(column),
                    )
                )
            continue
        else:
            column = str(entry)

        dynamic = column == "coded_passages"
        label_source = label_overrides.get(column) or label_hint
        description_hint = descriptions.get(column)
        if label_source:
            pretty = label_source
        else:
            pretty = column.replace("_", " ").title()
        identity = (column, dynamic)
        if identity in seen:
            continue
        seen.add(identity)
        requests.append(
            _AttributeRequest(
                column=column,
                label=pretty,
                dynamic=dynamic,
                description=description_hint,
            )
        )

    return requests


def _collect_mapping_keys(series: pd.Series, limit: int = 64) -> List[str]:
    keys: List[str] = []
    seen: Set[str] = set()
    for value in series:
        parsed = _parse_structured_cell(value)
        if not isinstance(parsed, Mapping):
            continue
        for key in parsed.keys():
            if key is None:
                continue
            text = str(key).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            keys.append(text)
            if len(keys) >= limit:
                return keys
    return keys


def _extract_mapping_value(value: Any, key: str) -> Any:
    parsed = _parse_structured_cell(value)
    if not isinstance(parsed, Mapping):
        return None
    raw = parsed.get(key)
    if isinstance(raw, Mapping):
        for candidate in ("value", "rating", "score", "answer", "label", "text"):
            if candidate in raw and raw[candidate] is not None:
                return raw[candidate]
        return raw.get("value")
    return raw


def _expand_mapping_attribute_requests(
    df: pd.DataFrame,
    requests: List[_AttributeRequest],
) -> Tuple[pd.DataFrame, List[_AttributeRequest]]:
    if df.empty or not requests:
        return df, requests

    expanded: List[_AttributeRequest] = []
    for request in requests:
        if request.dynamic or request.column not in df.columns:
            expanded.append(request)
            continue
        series = df[request.column]
        mapping_keys = _collect_mapping_keys(series)
        if not mapping_keys:
            expanded.append(request)
            continue
        for key in mapping_keys:
            derived_column = f"{request.column}::{key}"
            if derived_column not in df.columns:
                df[derived_column] = series.apply(
                    lambda value, key=key: _extract_mapping_value(value, key)
                )
            expanded.append(
                _AttributeRequest(
                    column=derived_column,
                    label=str(key),
                    dynamic=False,
                    description=request.description,
                )
            )
    return df, expanded


def _coerce_bool_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "y"}:
            return True
        if lowered in {"false", "no", "n"}:
            return False
    return None


def _coerce_numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _looks_like_snippet_column(values: Sequence[Any]) -> bool:
    hits = 0
    inspected = 0
    for value in values:
        if _is_na(value):
            continue
        inspected += 1
        parsed = _parse_structured_cell(value)
        if isinstance(parsed, dict) and parsed:
            hits += 1
            continue
        if isinstance(parsed, (list, tuple, set)):
            if any(str(item).strip() for item in parsed if not _is_na(item)):
                hits += 1
                continue
        if isinstance(parsed, str):
            candidate = _parse_structured_cell(parsed)
            if isinstance(candidate, (list, tuple, set)) and any(
                str(item).strip() for item in candidate if not _is_na(item)
            ):
                hits += 1
                continue
    if inspected == 0:
        return False
    return hits >= max(1, inspected // 2)


def _infer_attribute_kind(series: Optional[pd.Series]) -> Literal["snippet", "boolean", "numeric", "text"]:
    if series is None:
        return "text"

    sample: List[Any] = []
    for value in series:
        if _is_na(value):
            continue
        sample.append(value)
        if len(sample) >= 25:
            break

    if not sample:
        return "text"

    if _looks_like_snippet_column(sample):
        return "snippet"

    bool_hits = 0
    numeric_hits = 0
    for value in sample:
        if _coerce_bool_value(value) is not None:
            bool_hits += 1
            continue
        if _coerce_numeric_value(value) is not None:
            numeric_hits += 1

    threshold = max(1, int(len(sample) * 0.6))
    if bool_hits >= threshold:
        return "boolean"
    if numeric_hits >= threshold:
        return "numeric"
    return "text"


def _format_numeric_chip(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 100:
        text = f"{value:.0f}"
    elif magnitude >= 10:
        text = f"{value:.1f}"
    else:
        text = f"{value:.2f}"
    trimmed = text.rstrip("0").rstrip(".")
    if len(trimmed) > 4:
        trimmed = trimmed[:4]
    return trimmed


def _compute_slider_step(min_value: float, max_value: float) -> float:
    span = max_value - min_value
    if not math.isfinite(span) or span <= 0:
        return 0.1
    step = span / 200.0
    if step < 0.001:
        step = 0.001
    return step


def _passage_matches_filters(
    passage: Mapping[str, Any],
    *,
    required_snippets: Optional[Set[str]] = None,
    required_bools: Optional[Set[str]] = None,
    numeric_filters: Optional[Mapping[str, Tuple[float, float]]] = None,
) -> bool:
    """Return ``True`` if a passage matches the provided filter selections."""

    snippet_map = passage.get("snippets") or {}
    for category in required_snippets or ():
        if not snippet_map.get(category):
            return False

    bool_map = passage.get("bools") or {}
    for column in required_bools or ():
        if bool_map.get(column) is not True:
            return False

    numeric_map = passage.get("numeric") or {}
    for column, bounds in (numeric_filters or {}).items():
        value = numeric_map.get(column)
        if value is None:
            return False
        lower, upper = bounds
        if value < lower - 1e-9 or value > upper + 1e-9:
            return False

    return True


class PassageViewer:
    """Legacy desktop viewer placeholder."""

    def __init__(self, *_, **__):
        raise RuntimeError(
            "The tkinter-based PassageViewer has been retired. "
            "Use gabriel.view(...) inside a notebook environment."
        )

    def show(self):
        raise RuntimeError(
            "gabriel.view now renders exclusively via the notebook interface; "
            "desktop mode is no longer available."
        )

_COLAB_STYLE = """
<style>
.gabriel-codify-viewer {
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: #f5f7fa;
    background: transparent;
}
.gabriel-codify-viewer .gabriel-status {
    font-size: 14px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.7);
    margin-bottom: 8px;
}
.gabriel-codify-viewer .gabriel-controls {
    display: flex;
    gap: 10px;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 12px;
}
.gabriel-codify-viewer .gabriel-passage-panel {
    background: #13161a;
    border: 1px solid #2b323c;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 16px 40px rgba(9, 11, 16, 0.45);
}
.gabriel-codify-viewer .gabriel-passage-scroll {
    max-height: 560px;
    overflow-y: auto;
    padding-right: 12px;
}
.gabriel-codify-viewer .gabriel-legend {
    position: sticky;
    top: 0;
    z-index: 2;
    background: #13161a;
    padding-bottom: 12px;
    margin-bottom: 16px;
    border-bottom: 1px solid #2b323c;
}
.gabriel-codify-viewer .gabriel-legend-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
}
.gabriel-codify-viewer .gabriel-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 12px;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
    font-size: 13px;
    color: rgba(255, 255, 255, 0.88);
    cursor: pointer;
    transition: background 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
    text-decoration: none;
    font: inherit;
    line-height: 1.2;
}
.gabriel-codify-viewer .gabriel-legend-item:hover {
    background: rgba(255, 255, 255, 0.12);
    border-color: rgba(255, 255, 255, 0.18);
    transform: translateY(-1px);
}
.gabriel-codify-viewer .gabriel-legend-item:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.65);
}
.gabriel-codify-viewer .gabriel-legend-item span {
    pointer-events: none;
}
.gabriel-codify-viewer .gabriel-legend-color {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.18);
}
.gabriel-codify-viewer .gabriel-legend-label {
    font-weight: 600;
    color: inherit;
}
.gabriel-codify-viewer .gabriel-legend-count {
    font-size: 11px;
    padding: 2px 6px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.78);
}
.gabriel-codify-viewer .gabriel-header {
    margin-bottom: 14px;
    padding: 14px 16px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.gabriel-codify-viewer .gabriel-header-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
}
.gabriel-codify-viewer .gabriel-header-label {
    font-weight: 600;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: rgba(255, 255, 255, 0.7);
}
.gabriel-codify-viewer .gabriel-header-value {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.94);
}
.gabriel-codify-viewer .gabriel-active-cats {
    margin-top: 4px;
    font-size: 13px;
    color: rgba(255, 255, 255, 0.85);
}
.gabriel-codify-viewer .gabriel-chip-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 10px;
}
.gabriel-codify-viewer .gabriel-filter-panel {
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
    border-radius: 14px;
    padding: 12px 16px 10px;
    margin-bottom: 12px;
}
.gabriel-codify-viewer .gabriel-filter-groups {
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.gabriel-codify-viewer .gabriel-filter-group {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.gabriel-codify-viewer .gabriel-filter-title {
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: rgba(255, 255, 255, 0.65);
}
.gabriel-codify-viewer .gabriel-filter-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.gabriel-codify-viewer .gabriel-filter-toggle {
    border-radius: 999px;
    border: 1px solid rgba(255, 255, 255, 0.2);
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.88);
    padding: 4px 14px;
    font-size: 12px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.18s ease;
}
.gabriel-codify-viewer .gabriel-filter-toggle:hover {
    background: rgba(0, 188, 212, 0.18);
    border-color: rgba(0, 188, 212, 0.5);
}
.gabriel-codify-viewer .gabriel-filter-toggle[aria-pressed="true"] {
    background: rgba(0, 188, 212, 0.28);
    border-color: rgba(0, 188, 212, 0.7);
    color: #d7fbff;
    box-shadow: 0 6px 14px rgba(0, 188, 212, 0.22);
}
.gabriel-codify-viewer .gabriel-filter-toggle:focus-visible {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.45);
}
.gabriel-codify-viewer .gabriel-filter-slider {
    width: 100%;
    margin-top: 2px;
}
.gabriel-codify-viewer .gabriel-filter-note {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.68);
    margin-bottom: 6px;
}
.gabriel-codify-viewer .gabriel-filter-actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 6px;
}
.gabriel-codify-viewer .gabriel-filter-clear {
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: rgba(255, 255, 255, 0.06);
    color: rgba(255, 255, 255, 0.85);
    padding: 3px 12px;
    cursor: pointer;
    transition: background 0.2s ease, border-color 0.2s ease;
}
.gabriel-codify-viewer .gabriel-filter-clear:hover {
    background: rgba(0, 188, 212, 0.25);
    border-color: rgba(0, 188, 212, 0.6);
    color: #e8fbff;
}
.gabriel-codify-viewer .gabriel-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 12px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.88);
}
.gabriel-codify-viewer .gabriel-chip-label {
    font-weight: 600;
    color: inherit;
}
.gabriel-codify-viewer .gabriel-chip-value {
    font-weight: 700;
    background: rgba(255, 255, 255, 0.15);
    color: rgba(255, 255, 255, 0.95);
    padding: 0 8px;
    border-radius: 8px;
    font-size: 12px;
}
.gabriel-codify-viewer .gabriel-chip--boolean.gabriel-chip-state-true {
    background: rgba(0, 188, 212, 0.2);
    border-color: rgba(0, 188, 212, 0.5);
    color: #7ce8ff;
}
.gabriel-codify-viewer .gabriel-chip--boolean.gabriel-chip-state-false {
    opacity: 0.4;
}
.gabriel-codify-viewer .gabriel-chip--boolean.gabriel-chip-state-unknown {
    opacity: 0.3;
}
.gabriel-codify-viewer .gabriel-chip--numeric .gabriel-chip-value {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.96), rgba(235, 240, 255, 0.9));
    color: #0b0d11;
    border-radius: 8px;
    border: 1px solid rgba(9, 12, 16, 0.35);
    padding: 4px 12px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    letter-spacing: 0.04em;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    font-variant-numeric: tabular-nums;
}
.gabriel-codify-viewer .gabriel-note {
    font-size: 13px;
    color: rgba(255, 255, 255, 0.72);
    margin-bottom: 10px;
}
.gabriel-codify-viewer.gabriel-theme-dark {
    color-scheme: dark;
}
.gabriel-codify-viewer .gabriel-text {
    font-size: 15px;
    line-height: 1.7;
    color: rgba(245, 247, 250, 0.96);
}
.gabriel-codify-viewer .gabriel-text p {
    margin: 0 0 1em 0;
}
.gabriel-codify-viewer .gabriel-snippet {
    position: relative;
    border-radius: 6px;
    padding: 1px 5px;
    font-weight: 600;
    color: #0d1014;
    cursor: pointer;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.gabriel-codify-viewer .gabriel-snippet::after {
    content: attr(data-label);
    position: absolute;
    left: 0;
    bottom: 100%;
    transform: translateY(-6px);
    background: rgba(8, 11, 17, 0.92);
    color: #f8fafc;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 11px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease, transform 0.2s ease;
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.35);
    z-index: 5;
}
.gabriel-codify-viewer .gabriel-snippet:hover::after {
    opacity: 1;
    transform: translateY(-10px);
}
.gabriel-codify-viewer .gabriel-snippet-active {
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.8), 0 0 18px rgba(255, 255, 255, 0.35);
}
.gabriel-codify-viewer .gabriel-empty {
    font-style: italic;
    color: rgba(255, 255, 255, 0.65);
}
@media (prefers-color-scheme: light) {
    .gabriel-codify-viewer:not(.gabriel-theme-dark) {
        color: #1f2933;
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-passage-panel {
        background: #f7f9fb;
        border-color: #d0d7e2;
        box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-legend {
        background: #f7f9fb;
        border-color: #d0d7e2;
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-header {
        background: rgba(15, 23, 42, 0.06);
        border-color: rgba(15, 23, 42, 0.12);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-header-label {
        color: rgba(15, 23, 42, 0.65);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-header-value {
        color: rgba(15, 23, 42, 0.92);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-text {
        color: #1f2933;
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-legend-item {
        background: rgba(15, 23, 42, 0.06);
        border-color: rgba(15, 23, 42, 0.12);
        color: rgba(15, 23, 42, 0.82);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-legend-item:hover {
        background: rgba(15, 23, 42, 0.1);
        border-color: rgba(15, 23, 42, 0.18);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-legend-item:focus-visible {
        box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.4);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-legend-count {
        background: rgba(15, 23, 42, 0.1);
        color: rgba(15, 23, 42, 0.75);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-chip {
        background: rgba(15, 23, 42, 0.05);
        border-color: rgba(15, 23, 42, 0.12);
        color: rgba(15, 23, 42, 0.8);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-chip-value {
        background: rgba(15, 23, 42, 0.08);
        color: rgba(15, 23, 42, 0.85);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-panel {
        background: rgba(15, 23, 42, 0.04);
        border-color: rgba(15, 23, 42, 0.12);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-title {
        color: rgba(15, 23, 42, 0.6);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-toggle {
        background: rgba(15, 23, 42, 0.05);
        border-color: rgba(15, 23, 42, 0.2);
        color: rgba(15, 23, 42, 0.78);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-toggle:hover {
        background: rgba(0, 188, 212, 0.22);
        color: rgba(15, 23, 42, 0.95);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-toggle[aria-pressed="true"] {
        color: rgba(15, 23, 42, 0.98);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-clear {
        background: rgba(15, 23, 42, 0.05);
        border-color: rgba(15, 23, 42, 0.16);
        color: rgba(15, 23, 42, 0.7);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-clear:hover {
        color: rgba(15, 23, 42, 0.95);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-filter-note {
        color: rgba(15, 23, 42, 0.6);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-chip--numeric .gabriel-chip-value {
        background: linear-gradient(135deg, #ffffff, #edf2ff);
        border-color: rgba(15, 23, 42, 0.2);
        color: rgba(15, 23, 42, 0.95);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.18);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-note {
        color: rgba(15, 23, 42, 0.6);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-empty {
        color: rgba(15, 23, 42, 0.55);
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-snippet::after {
        background: rgba(15, 23, 42, 0.92);
        color: #f8fafc;
    }
    .gabriel-codify-viewer:not(.gabriel-theme-dark) .gabriel-snippet-active {
        box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.28), 0 0 18px rgba(15, 23, 42, 0.3);
    }
}
.gabriel-codify-viewer.gabriel-theme-light {
    color: #1f2933;
    color-scheme: light;
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-passage-panel {
    background: #f7f9fb;
    border-color: #d0d7e2;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.12);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-legend {
    background: #f7f9fb;
    border-color: #d0d7e2;
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-header {
    background: rgba(15, 23, 42, 0.06);
    border-color: rgba(15, 23, 42, 0.12);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-header-label {
    color: rgba(15, 23, 42, 0.65);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-header-value {
    color: rgba(15, 23, 42, 0.92);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-text {
    color: #1f2933;
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-legend-item {
    background: rgba(15, 23, 42, 0.06);
    border-color: rgba(15, 23, 42, 0.12);
    color: rgba(15, 23, 42, 0.82);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-legend-item:hover {
    background: rgba(15, 23, 42, 0.1);
    border-color: rgba(15, 23, 42, 0.18);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-legend-item:focus-visible {
    box-shadow: 0 0 0 2px rgba(0, 188, 212, 0.4);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-legend-count {
    background: rgba(15, 23, 42, 0.1);
    color: rgba(15, 23, 42, 0.75);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-chip {
    background: rgba(15, 23, 42, 0.05);
    border-color: rgba(15, 23, 42, 0.12);
    color: rgba(15, 23, 42, 0.8);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-chip-value {
    background: rgba(15, 23, 42, 0.08);
    color: rgba(15, 23, 42, 0.85);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-chip--numeric .gabriel-chip-value {
    background: linear-gradient(135deg, #ffffff, #eef3ff);
    border-color: rgba(15, 23, 42, 0.18);
    color: rgba(15, 23, 42, 0.9);
    box-shadow: 0 8px 20px rgba(15, 23, 42, 0.16);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-note {
    color: rgba(15, 23, 42, 0.6);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-empty {
    color: rgba(15, 23, 42, 0.55);
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-snippet::after {
    background: rgba(15, 23, 42, 0.92);
    color: #f8fafc;
}
.gabriel-codify-viewer.gabriel-theme-light .gabriel-snippet-active {
    box-shadow: 0 0 0 2px rgba(15, 23, 42, 0.28), 0 0 18px rgba(15, 23, 42, 0.3);
}
</style>
<script>
(function () {
    if (window.__gabrielPassageViewerEnhancer) {
        return;
    }
    window.__gabrielPassageViewerEnhancer = true;

    const stateMap = new WeakMap();

    function ensureState(container, token) {
        let record = stateMap.get(container);
        if (!record || record.token !== token) {
            record = { token: token, indices: {} };
            stateMap.set(container, record);
        }
        return record;
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/[^a-zA-Z0-9_-]/g, '\\$&');
    }

    function bindLegendItem(item) {
        if (!(item instanceof Element) || item.dataset.gabrielBound === '1') {
            return;
        }
        const legend = item.closest('.gabriel-legend');
        const container = item.closest('.gabriel-codify-viewer');
        if (!legend || !container) {
            return;
        }
        const category = item.getAttribute('data-category');
        if (!category) {
            return;
        }
        const token = legend.getAttribute('data-legend-token') || '';
        const state = ensureState(container, token);
        item.dataset.gabrielBound = '1';
        item.addEventListener('click', function (event) {
            event.preventDefault();
            const selector = '.gabriel-snippet[data-category="' + escapeSelector(category) + '"]';
            const snippets = container.querySelectorAll(selector);
            if (!snippets.length) {
                return;
            }
            const nextIndex = state.indices[category] || 0;
            const target = snippets[nextIndex % snippets.length];
            state.indices[category] = (nextIndex + 1) % snippets.length;
            container.querySelectorAll('.gabriel-snippet.gabriel-snippet-active').forEach(function (el) {
                if (el !== target) {
                    el.classList.remove('gabriel-snippet-active');
                }
            });
            target.classList.add('gabriel-snippet-active');
            if (typeof target.scrollIntoView === 'function') {
                target.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            window.setTimeout(function () {
                target.classList.remove('gabriel-snippet-active');
            }, 1600);
        });
    }

    function scan(root) {
        if (!(root instanceof Element)) {
            return;
        }
        root.querySelectorAll('.gabriel-legend-item').forEach(bindLegendItem);
    }

    const observer = new MutationObserver(function (mutations) {
        mutations.forEach(function (mutation) {
            mutation.addedNodes.forEach(function (node) {
                if (!(node instanceof Element)) {
                    return;
                }
                if (node.classList.contains('gabriel-legend-item')) {
                    bindLegendItem(node);
                } else {
                    scan(node);
                }
            });
        });
    });

    if (document && document.body) {
        observer.observe(document.body, { childList: true, subtree: true });
        scan(document.body);
    }
})();
</script>
"""

def _build_style_overrides(
    font_scale: float = 1.0,
    font_family: Optional[str] = None,
    color_mode: str = "auto",
) -> str:
    """Return additional CSS overrides based on user preferences."""

    fragments: List[str] = []

    try:
        scale = float(font_scale)
    except (TypeError, ValueError):
        scale = 1.0
    if not math.isfinite(scale):
        scale = 1.0
    scale = max(0.6, min(2.5, scale))

    if not math.isclose(scale, 1.0, rel_tol=1e-3):
        for selector, base_size in _FONT_SIZE_OVERRIDES.items():
            scaled = max(8.0, min(48.0, base_size * scale))
            fragments.append(f"{selector} {{ font-size: {scaled:.2f}px; }}")

    if font_family:
        preferred = str(font_family).strip()
        if preferred:
            sanitized = preferred.replace("'", "\\'")
            fragments.append(
                ".gabriel-codify-viewer { font-family: '"
                + sanitized
                + "', 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif; }"
            )

    if color_mode == "light":
        fragments.append(".gabriel-codify-viewer { color-scheme: light; }")
    elif color_mode == "dark":
        fragments.append(".gabriel-codify-viewer { color-scheme: dark; }")

    if not fragments:
        return ""

    return "<style>" + "".join(fragments) + "</style>"


def _normalize_header_columns(
    header_columns: Optional[Union[Sequence[Any], Any]]
) -> List[Tuple[str, str]]:
    if header_columns is None:
        return []

    if isinstance(header_columns, Mapping):
        header_sequence: Iterable[Any] = header_columns.items()
    elif isinstance(header_columns, (str, bytes)):
        header_sequence = [header_columns]
    elif isinstance(header_columns, Iterable):
        header_sequence = header_columns
    else:
        header_sequence = [header_columns]

    normalized: List[Tuple[str, str]] = []
    for entry in header_sequence:
        if isinstance(entry, tuple) and entry:
            column = str(entry[0])
            label = str(entry[1]) if len(entry) > 1 else column
        elif isinstance(entry, list) and entry:
            column = str(entry[0])
            label = str(entry[1]) if len(entry) > 1 else column
        else:
            column = str(entry)
            label = column
        pretty_label = label.replace("_", " ").title()
        normalized.append((column, pretty_label))
    return normalized


def _is_na(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    return False


def _parse_structured_cell(value: Any) -> Any:
    """Attempt to parse serialized list/dict values from CSV/JSON sources."""

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return []

    lowered = stripped.lower()
    if lowered in {"nan", "none", "null", "n/a"}:
        return None

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(stripped)
        except Exception:
            continue

    return value


def _coerce_snippet_list(value: Any) -> List[str]:
    parsed = _parse_structured_cell(value)

    if _is_na(parsed) or parsed is None:
        return []

    if isinstance(parsed, str):
        return [parsed] if parsed.strip() else []

    if isinstance(parsed, (list, tuple, set)):
        snippets: List[str] = []
        for item in parsed:
            if _is_na(item) or item is None:
                continue
            text = str(item)
            if text.strip():
                snippets.append(text)
        return snippets

    text = str(parsed)
    return [text] if text.strip() else []


def _coerce_coded_passage_map(value: Any) -> Dict[str, List[str]]:
    parsed = _parse_structured_cell(value)

    if _is_na(parsed) or parsed is None:
        return {}

    if isinstance(parsed, dict):
        normalized: Dict[str, List[str]] = {}
        for key, snippets in parsed.items():
            if _is_na(key) or key is None:
                continue
            cat = str(key)
            normalized[cat] = _coerce_snippet_list(snippets)
        return normalized

    if isinstance(parsed, (list, tuple)):
        aggregated: Dict[str, List[str]] = {}
        for entry in parsed:
            if isinstance(entry, dict):
                for key, snippets in entry.items():
                    cat = str(key)
                    aggregated.setdefault(cat, []).extend(_coerce_snippet_list(snippets))
        return aggregated

    return {}


def _normalize_structured_dataframe(
    df: pd.DataFrame,
    categories: Optional[Union[List[str], str]],
) -> pd.DataFrame:
    if "coded_passages" in df.columns:
        df["coded_passages"] = df["coded_passages"].apply(_coerce_coded_passage_map)

    category_columns: Iterable[str]
    if categories is None or categories == "coded_passages":
        category_columns = []
    else:
        category_columns = categories

    for column in category_columns:
        if column in df.columns:
            df[column] = df[column].apply(_coerce_snippet_list)

    return df


def _extract_categories_from_coded_passages(df: pd.DataFrame) -> List[str]:
    if "coded_passages" not in df.columns:
        return []

    all_categories = set()
    for entry in df["coded_passages"]:
        if isinstance(entry, dict):
            for key in entry.keys():
                all_categories.add(str(key))

    return sorted(all_categories)


def _format_header_value(value: Any) -> str:
    if _is_na(value):
        return ""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    return str(value)


def _build_highlighted_text(
    text: str,
    snippet_map: Dict[str, List[str]],
    category_colors: Dict[str, str],
    category_labels: Dict[str, str],
) -> str:
    if not text:
        return "<div class='gabriel-empty'>No text available.</div>"

    spans: List[Tuple[int, int, str]] = []
    for category, snippets in snippet_map.items():
        if not snippets or category not in category_colors:
            continue
        for snippet in snippets:
            if not snippet:
                continue
            start = 0
            while True:
                index = text.find(snippet, start)
                if index == -1:
                    break
                spans.append((index, index + len(snippet), category))
                start = index + len(snippet)

    if not spans:
        return html.escape(text).replace("\n", "<br/>")

    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    merged: List[Tuple[int, int, str]] = []
    current_end = -1
    for start, end, category in spans:
        if start < current_end:
            continue
        merged.append((start, end, category))
        current_end = end

    pieces: List[str] = []
    cursor = 0
    snippet_indices: Dict[str, int] = {}
    for start, end, category in merged:
        pieces.append(html.escape(text[cursor:start]).replace("\n", "<br/>"))
        snippet_html = html.escape(text[start:end]).replace("\n", "<br/>")
        category_key = str(category)
        label_source = category_labels.get(category_key, category_key)
        label = html.escape(label_source.replace("_", " ").title())
        color = category_colors.get(category_key, "#ffd54f")
        safe_color = html.escape(color, quote=True)
        safe_category = html.escape(category_key, quote=True)
        index = snippet_indices.get(category_key, 0)
        snippet_indices[category_key] = index + 1
        slug = re.sub(r"[^0-9a-zA-Z_-]+", "-", category_key).strip("-")
        if not slug:
            slug = "category"
        element_id = f"gabriel-snippet-{slug}-{index}"
        pieces.append(
            "<span class='gabriel-snippet' "
            f"data-category='{safe_category}' data-index='{index}' "
            f"data-label='{label}' id='{element_id}' "
            f"style='background-color:{safe_color}' title='{label}'>"
            f"{snippet_html}</span>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]).replace("\n", "<br/>"))
    return "".join(pieces)


def _build_header_html(
    header_rows: List[Tuple[str, str]],
    active_categories: List[str],
    chips: Optional[List[Dict[str, str]]] = None,
) -> str:
    if not header_rows and not active_categories and not chips:
        return ""

    parts: List[str] = []
    if chips:
        parts.append(_build_chip_html(chips))
    for label, value in header_rows:
        safe_label = html.escape(label)
        safe_value = html.escape(value).replace("\n", "<br/>")
        parts.append(
            f"<div class='gabriel-header-row'>"
            f"<span class='gabriel-header-label'>{safe_label}:</span>"
            f"<span class='gabriel-header-value'>{safe_value}</span>"
            f"</div>"
        )

    if active_categories:
        active = ", ".join(
            html.escape(cat.replace("_", " ").title()) for cat in active_categories
        )
        parts.append(
            "<div class='gabriel-active-cats'><strong>Categories:</strong> "
            f"{active}</div>"
        )

    return "<div class='gabriel-header'>" + "".join(parts) + "</div>"


def _build_chip_html(chips: Optional[List[Dict[str, str]]]) -> str:
    if not chips:
        return ""

    items: List[str] = ["<div class='gabriel-chip-grid'>"]
    for chip in chips:
        label = html.escape(chip.get("label", ""))
        value = chip.get("value")
        safe_value = html.escape(value, quote=False) if value else ""
        kind = chip.get("kind", "generic")
        state = chip.get("state", "")
        title = chip.get("title") or chip.get("label", "")
        safe_title = html.escape(title, quote=True)
        class_names = ["gabriel-chip", f"gabriel-chip--{kind}"]
        if state:
            class_names.append(f"gabriel-chip-state-{state}")
        safe_state = html.escape(state, quote=True)
        items.append(
            f"<div class='{' '.join(class_names)}' data-kind='{kind}' "
            f"data-state='{safe_state}' title='{safe_title}'>"
            f"<span class='gabriel-chip-label'>{label}</span>"
        )
        if safe_value:
            items.append(f"<span class='gabriel-chip-value'>{safe_value}</span>")
        items.append("</div>")
    items.append("</div>")
    return "".join(items)


def _build_note_html(notes: Optional[Sequence[Any]]) -> str:
    """Render helper notes such as truncation or text-only warnings."""

    if not notes:
        return ""

    fragments: List[str] = []
    for message in notes:
        if message is None:
            continue
        text = str(message).strip()
        if not text:
            continue
        fragments.append(
            f"<div class='gabriel-note'>{html.escape(text)}</div>"
        )

    return "".join(fragments)


def _build_legend_html(
    category_colors: Dict[str, str],
    category_counts: Dict[str, int],
    category_labels: Dict[str, str],
    legend_token: Optional[str] = None,
) -> str:
    if not category_colors:
        return ""

    items = []
    for category, color in category_colors.items():
        pretty = category_labels.get(category, category).replace("_", " ").title()
        label = html.escape(pretty)
        raw_count = category_counts.get(category, 0)
        try:
            count_value = int(raw_count)
        except (TypeError, ValueError):
            count_value = 0
        aria_label = html.escape(f"{pretty} ({count_value})", quote=True)
        count = html.escape(str(count_value))
        safe_color = html.escape(color, quote=True)
        safe_category = html.escape(category, quote=True)
        items.append(
            "<button type='button' class='gabriel-legend-item' "
            f"data-category='{safe_category}' data-count='{count}' aria-label='{aria_label}'>"
            f"<span class='gabriel-legend-color' style='background:{safe_color}'></span>"
            f"<span class='gabriel-legend-label'>{label}</span>"
            f"<span class='gabriel-legend-count'>{count}</span>"
            "</button>"
        )

    token_attr = (
        f" data-legend-token='{html.escape(legend_token, quote=True)}'"
        if legend_token
        else ""
    )
    return (
        f"<div class='gabriel-legend'{token_attr}><div class='gabriel-legend-grid'>"
        + "".join(items)
        + "</div></div>"
    )


def _render_passage_viewer(
    df: pd.DataFrame,
    column_name: str,
    attributes: Optional[Union[Mapping[str, Any], Sequence[Any], Any]] = None,
    header_columns: Optional[Union[Sequence[Any], Any]] = None,
    *,
    max_passages: Optional[int] = None,
    font_scale: float = 1.0,
    font_family: Optional[str] = None,
    color_mode: str = "auto",
) -> None:
    """Display passages inside a Jupyter notebook."""

    from IPython.display import HTML, display  # pragma: no cover - optional

    df = df.copy()
    attribute_requests = _normalize_attribute_requests(attributes)
    if not attribute_requests and "coded_passages" in df.columns:
        attribute_requests = [
            _AttributeRequest("coded_passages", "Coded Passages", dynamic=True)
        ]
    df, attribute_requests = _expand_mapping_attribute_requests(df, attribute_requests)

    attribute_specs: List[_AttributeSpec] = []
    boolean_specs: List[_AttributeSpec] = []
    numeric_specs: List[_AttributeSpec] = []
    snippet_columns: List[str] = []
    for request in attribute_requests:
        if request.dynamic:
            attribute_specs.append(
                _AttributeSpec(
                    request.column,
                    request.label,
                    "snippet",
                    dynamic=True,
                    description=request.description,
                )
            )
            continue
        series = df[request.column] if request.column in df.columns else None
        kind = _infer_attribute_kind(series)
        spec = _AttributeSpec(
            request.column,
            request.label,
            kind,
            dynamic=False,
            description=request.description,
        )
        attribute_specs.append(spec)
        if kind == "snippet":
            snippet_columns.append(request.column)
        elif kind == "boolean":
            boolean_specs.append(spec)
        elif kind == "numeric":
            numeric_specs.append(spec)

    df = _normalize_structured_dataframe(df, snippet_columns)
    normalized_headers = _normalize_header_columns(header_columns)

    has_dynamic = any(spec.dynamic for spec in attribute_specs)
    text_only_mode = not attribute_specs
    category_names: List[str] = []
    category_labels: Dict[str, str] = {}
    if has_dynamic:
        for cat in _extract_categories_from_coded_passages(df):
            if cat not in category_labels:
                category_names.append(cat)
                category_labels[cat] = cat
    for spec in attribute_specs:
        if spec.kind == "snippet" and not spec.dynamic:
            if spec.column not in category_labels:
                category_names.append(spec.column)
                category_labels[spec.column] = spec.label

    colors = _generate_distinct_colors(len(category_names))
    category_colors = dict(zip(category_names, colors))

    passages: List[Dict[str, Any]] = []
    numeric_ranges: Dict[str, Tuple[float, float]] = {}
    for _, row in df.iterrows():
        raw_text = row.get(column_name)
        text = "" if _is_na(raw_text) else str(raw_text)
        snippet_map: Dict[str, List[str]] = {cat: [] for cat in category_names}
        bool_values: Dict[str, Optional[bool]] = {}
        numeric_values: Dict[str, Optional[float]] = {}
        if has_dynamic:
            raw_map = row.get("coded_passages")
            if isinstance(raw_map, dict):
                for cat, snippets in raw_map.items():
                    cat_key = str(cat)
                    if cat_key in snippet_map:
                        snippet_map[cat_key] = _coerce_snippet_list(snippets)

        for spec in attribute_specs:
            if spec.dynamic or spec.kind != "snippet":
                continue
            cat_key = spec.column
            if cat_key in snippet_map:
                snippet_map[cat_key] = _coerce_snippet_list(row.get(cat_key, []))

        header_rows: List[Tuple[str, str]] = []
        for column, label in normalized_headers:
            value = row.get(column)
            formatted = _format_header_value(value)
            if formatted:
                header_rows.append((label, formatted))

        chips: List[Dict[str, str]] = []
        text_attributes: List[Tuple[str, str]] = []
        for spec in attribute_specs:
            if spec.dynamic or spec.kind == "snippet":
                continue
            value = row.get(spec.column)
            if spec.kind == "boolean":
                bool_value = _coerce_bool_value(value)
                bool_values[spec.column] = bool_value
                if bool_value is True:
                    display_val = "True"
                    state = "true"
                elif bool_value is False:
                    display_val = "False"
                    state = "false"
                else:
                    display_val = "—"
                    state = "unknown"
                chips.append(
                    {
                        "label": spec.label,
                        "value": display_val,
                        "state": state,
                        "kind": "boolean",
                        "title": spec.description,
                    }
                )
            elif spec.kind == "numeric":
                numeric_value = _coerce_numeric_value(value)
                numeric_values[spec.column] = numeric_value
                if numeric_value is not None:
                    if math.isfinite(numeric_value):
                        existing = numeric_ranges.get(spec.column)
                        if existing is None:
                            numeric_ranges[spec.column] = (
                                numeric_value,
                                numeric_value,
                            )
                        else:
                            numeric_ranges[spec.column] = (
                                min(existing[0], numeric_value),
                                max(existing[1], numeric_value),
                            )
                    chips.append(
                        {
                            "label": spec.label,
                            "value": _format_numeric_chip(numeric_value),
                            "state": "number",
                            "kind": "numeric",
                            "title": spec.description,
                        }
                    )
                else:
                    formatted = _format_header_value(value)
                    if formatted:
                        text_attributes.append((spec.label, formatted))
            else:
                formatted = _format_header_value(value)
                if formatted:
                    text_attributes.append((spec.label, formatted))

        if text_attributes:
            header_rows.extend(text_attributes)

        active_categories = [cat for cat, snippets in snippet_map.items() if snippets]
        passage_counts = {
            cat: len(snippet_map.get(cat, []))
            for cat in category_names
        }
        passages.append(
            {
                "text": text,
                "snippets": snippet_map,
                "header": header_rows,
                "active": active_categories,
                "counts": passage_counts,
                "chips": chips,
                "bools": bool_values,
                "numeric": numeric_values,
            }
        )

    color_choice = str(color_mode or "auto").lower()
    if color_choice not in {"auto", "dark", "light"}:
        color_choice = "auto"
    theme_class = (
        " gabriel-theme-light"
        if color_choice == "light"
        else (" gabriel-theme-dark" if color_choice == "dark" else "")
    )

    style_html = _COLAB_STYLE + _build_style_overrides(
        font_scale=font_scale,
        font_family=font_family,
        color_mode=color_choice,
    )

    original_total = len(passages)
    limit = max_passages
    trunc_note: Optional[str] = None
    if limit is not None and limit >= 0 and original_total > limit:
        trunc_note = f"Showing first {limit} of {original_total} passages."
        passages = passages[:limit]

    total = len(passages)
    note_messages: List[str] = []
    if text_only_mode:
        note_messages.append(
            "Displaying raw text because no attribute columns were provided."
        )
    if trunc_note:
        note_messages.append(trunc_note)
    note_html = _build_note_html(note_messages)
    root_class = f"gabriel-codify-viewer{theme_class}"

    viewer_id = f"gabriel-viewer-{uuid.uuid4().hex}"
    if total == 0:
        empty_html = (
            f"<div class='{root_class}'><div class='gabriel-empty'>No passages to display.</div>"
            f"{note_html}</div>"
        )
        display(HTML(style_html + empty_html))
        return

    render_entries: List[Dict[str, Any]] = []
    for idx, payload in enumerate(passages):
        body_html = _build_highlighted_text(
            payload["text"], payload["snippets"], category_colors, category_labels
        )
        header_html = _build_header_html(
            payload["header"], payload["active"], payload.get("chips")
        )
        legend_token = f"interactive-{idx}-{random.random()}"
        legend_html = _build_legend_html(
            category_colors, payload["counts"], category_labels, legend_token
        )
        snippet_flags = {
            cat: bool(payload["snippets"].get(cat)) for cat in category_names
        }
        bool_map: Dict[str, Optional[bool]] = {}
        for column, value in (payload.get("bools") or {}).items():
            if value is True:
                bool_map[column] = True
            elif value is False:
                bool_map[column] = False
            else:
                bool_map[column] = None
        numeric_map: Dict[str, Optional[float]] = {}
        for column, value in (payload.get("numeric") or {}).items():
            if value is None:
                numeric_map[column] = None
            else:
                try:
                    numeric_map[column] = float(value)
                except (TypeError, ValueError):
                    numeric_map[column] = None
        render_entries.append(
            {
                "html": f"{legend_html}{header_html}<div class='gabriel-text'>{body_html}</div>",
                "snippets": snippet_flags,
                "bools": bool_map,
                "numeric": numeric_map,
            }
        )

    category_filter_defs = [
        {
            "id": cat,
            "label": category_labels.get(cat, cat).replace("_", " ").title(),
        }
        for cat in category_names
    ]
    boolean_filter_defs = [
        {
            "column": spec.column,
            "label": spec.label or spec.column.replace("_", " ").title(),
        }
        for spec in boolean_specs
    ]
    numeric_filter_defs: List[Dict[str, Any]] = []
    for spec in numeric_specs:
        bounds = numeric_ranges.get(spec.column)
        if not bounds:
            continue
        lower, upper = bounds
        if not (math.isfinite(lower) and math.isfinite(upper)):
            continue
        if upper <= lower:
            continue
        numeric_filter_defs.append(
            {
                "column": spec.column,
                "label": spec.label or spec.column.replace("_", " ").title(),
                "min": lower,
                "max": upper,
                "step": _compute_slider_step(lower, upper),
            }
        )

    has_filters = bool(
        category_filter_defs or boolean_filter_defs or numeric_filter_defs
    )
    data_payload = {
        "passages": render_entries,
        "categoryFilters": category_filter_defs,
        "booleanFilters": boolean_filter_defs,
        "numericFilters": numeric_filter_defs,
        "hasFilters": has_filters,
    }
    data_json = json.dumps(data_payload).replace("</", r"<\/")
    note_block = (
        f"<div class='gabriel-note-stack' data-role='note'>{note_html}</div>"
        if note_html
        else ""
    )
    slider_max = max(1, total)
    slider_count = f"1 / {total}"
    viewer_template = Template(
        """
<div id="$viewer_id" class="$root_class">
  <div class="gabriel-status" data-role="status"></div>
  <div class="gabriel-controls">
    <div class="gabriel-nav-group">
      <button type="button" class="gabriel-nav-button" data-action="prev">◀ Previous</button>
      <button type="button" class="gabriel-nav-button" data-action="random">Random</button>
      <button type="button" class="gabriel-nav-button" data-action="next">Next ▶</button>
    </div>
    <div class="gabriel-slider-shell">
      <input type="range" min="1" max="$slider_max" value="1" class="gabriel-slider" data-role="slider" />
      <div class="gabriel-slider-count" data-role="slider-count">$slider_count</div>
    </div>
  </div>
  <div class="gabriel-filter-panel" data-role="filter-panel">
    <div class="gabriel-filter-title">Filters</div>
    <div class="gabriel-filter-groups" data-role="filters"></div>
    <div class="gabriel-filter-actions">
      <button type="button" class="gabriel-filter-clear" data-role="clear-filters">Clear filters</button>
    </div>
    <div class="gabriel-filter-note" data-role="filter-note"></div>
  </div>
  <div class="gabriel-passage-panel">
    <div class="gabriel-passage-scroll" data-role="passage"></div>
  </div>
  $note_block
</div>
<script>
(function() {
    const container = document.getElementById("$viewer_id");
    if (!container) {
        return;
    }
    const data = $data_json;
    const statusEl = container.querySelector('[data-role="status"]');
    const passageEl = container.querySelector('[data-role="passage"]');
    const sliderEl = container.querySelector('[data-role="slider"]');
    const sliderCountEl = container.querySelector('[data-role="slider-count"]');
    const filterPanel = container.querySelector('[data-role="filter-panel"]');
    const filtersHost = container.querySelector('[data-role="filters"]');
    const filterNoteEl = container.querySelector('[data-role="filter-note"]');
    const clearFiltersBtn = container.querySelector('[data-role="clear-filters"]');
    const prevBtn = container.querySelector('[data-action="prev"]');
    const nextBtn = container.querySelector('[data-action="next"]');
    const randomBtn = container.querySelector('[data-action="random"]');
    const total = data.passages.length;
    if (!total) {
        if (passageEl) {
            passageEl.innerHTML = "<div class='gabriel-empty'>No passages to display.</div>";
        }
        if (statusEl) {
            statusEl.textContent = "No passages available.";
        }
        if (filterPanel) {
            filterPanel.style.display = "none";
        }
        if (sliderEl) sliderEl.disabled = true;
        if (prevBtn) prevBtn.disabled = true;
        if (nextBtn) nextBtn.disabled = true;
        if (randomBtn) randomBtn.disabled = true;
        return;
    }
    const state = {
        active: data.passages.map((_, idx) => idx),
        index: 0,
        snippets: new Set(),
        bools: new Set(),
        numeric: {},
    };
    const numericDefaults = {};
    const numericHandles = {};

    function formatNumber(value) {
        if (typeof value !== 'number' || !isFinite(value)) {
            return '—';
        }
        const abs = Math.abs(value);
        let text;
        if (abs >= 100) {
            text = value.toFixed(0);
        } else if (abs >= 10) {
            text = value.toFixed(1);
        } else {
            text = value.toFixed(2);
        }
        return text.replace(/\\.0+$$/, '').replace(/(\\.[0-9]*[1-9])0+$$/, '$$1');
    }

    function matchesFilters(entry) {
        if (!entry) {
            return false;
        }
        if (state.snippets.size) {
            for (const cat of state.snippets) {
                if (!entry.snippets || !entry.snippets[cat]) {
                    return false;
                }
            }
        }
        if (state.bools.size) {
            for (const column of state.bools) {
                if (!entry.bools || entry.bools[column] !== true) {
                    return false;
                }
            }
        }
        for (const [column, config] of Object.entries(state.numeric)) {
            if (!config || !config.active) {
                continue;
            }
            const value = entry.numeric ? entry.numeric[column] : null;
            if (typeof value !== 'number') {
                return false;
            }
            if (value < config.min - 1e-9 || value > config.max + 1e-9) {
                return false;
            }
        }
        return true;
    }

    function renderPassage() {
        if (!passageEl) {
            return;
        }
        if (!state.active.length) {
            passageEl.innerHTML = "<div class='gabriel-empty'>No passages match the current filters.</div>";
            return;
        }
        const idx = state.active[state.index];
        const payload = data.passages[idx];
        passageEl.innerHTML = payload.html;
    }

    function updateStatus() {
        if (!statusEl) {
            return;
        }
        if (!state.active.length) {
            statusEl.textContent = 'No passages match the current filters.';
            return;
        }
        statusEl.innerHTML = 'Passage <strong>' + (state.index + 1) + '</strong> of ' + state.active.length;
    }

    function updateSlider(forceValue) {
        if (!sliderEl || !sliderCountEl) {
            return;
        }
        const max = Math.max(1, state.active.length || 1);
        sliderEl.max = String(max);
        if (forceValue !== false) {
            sliderEl.value = state.active.length ? String(state.index + 1) : '1';
        }
        sliderEl.disabled = state.active.length <= 1;
        sliderCountEl.textContent = state.active.length ? (state.index + 1) + ' / ' + state.active.length : '0 / 0';
    }

    function updateFilterNote() {
        if (!filterNoteEl) {
            return;
        }
        if (!state.active.length) {
            filterNoteEl.textContent = '0 of ' + total + ' passages match filters.';
            return;
        }
        if (
            state.active.length === total &&
            !state.snippets.size &&
            !state.bools.size &&
            !Object.values(state.numeric).some(cfg => cfg && cfg.active)
        ) {
            filterNoteEl.textContent = '';
        } else {
            filterNoteEl.textContent = state.active.length + ' of ' + total + ' passages match filters.';
        }
    }

    function updateNavDisabled() {
        const disabled = state.active.length === 0;
        if (prevBtn) prevBtn.disabled = disabled;
        if (nextBtn) nextBtn.disabled = disabled;
        if (randomBtn) randomBtn.disabled = disabled;
    }

    function applyFilters(resetIndex = true) {
        const matches = [];
        data.passages.forEach((entry, idx) => {
            if (matchesFilters(entry)) {
                matches.push(idx);
            }
        });
        state.active = matches;
        if (!matches.length) {
            state.index = 0;
        } else if (resetIndex || state.index >= matches.length) {
            state.index = 0;
        }
        renderPassage();
        updateStatus();
        updateSlider();
        updateFilterNote();
        updateNavDisabled();
    }

    function moveIndex(delta) {
        if (!state.active.length) {
            return;
        }
        const length = state.active.length;
        state.index = (state.index + delta + length) % length;
        renderPassage();
        updateStatus();
        updateSlider();
    }

    function handleNumericChange(column) {
        const handle = numericHandles[column];
        if (!handle) {
            return;
        }
        let start = parseFloat(handle.minInput.value);
        let end = parseFloat(handle.maxInput.value);
        if (isNaN(start) || isNaN(end)) {
            return;
        }
        if (start > end) {
            const temp = start;
            start = end;
            end = temp;
        }
        handle.minInput.value = String(start);
        handle.maxInput.value = String(end);
        handle.minValue.textContent = formatNumber(start);
        handle.maxValue.textContent = formatNumber(end);
        const defaults = numericDefaults[column];
        const isActive = !defaults || Math.abs(start - defaults.min) > 1e-9 || Math.abs(end - defaults.max) > 1e-9;
        state.numeric[column] = { min: start, max: end, active: Boolean(isActive) };
        applyFilters(false);
    }

    function createToggleGroup(title, filters, kind) {
        const group = document.createElement('div');
        group.className = 'gabriel-filter-group';
        const heading = document.createElement('div');
        heading.className = 'gabriel-filter-title';
        heading.textContent = title;
        group.appendChild(heading);
        const row = document.createElement('div');
        row.className = 'gabriel-filter-row';
        filters.forEach(filter => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'gabriel-filter-toggle';
            button.setAttribute('aria-pressed', 'false');
            button.textContent = filter.label;
            if (kind === 'category') {
                button.dataset.category = filter.id;
            } else if (kind === 'boolean') {
                button.dataset.boolean = filter.column;
            }
            button.addEventListener('click', () => {
                const key = kind === 'category' ? button.dataset.category : button.dataset.boolean;
                if (!key) {
                    return;
                }
                const target = kind === 'category' ? state.snippets : state.bools;
                if (target.has(key)) {
                    target.delete(key);
                    button.setAttribute('aria-pressed', 'false');
                } else {
                    target.add(key);
                    button.setAttribute('aria-pressed', 'true');
                }
                applyFilters(true);
            });
            row.appendChild(button);
        });
        group.appendChild(row);
        return group;
    }

    function createNumericGroup(filters) {
        const group = document.createElement('div');
        group.className = 'gabriel-filter-group';
        const heading = document.createElement('div');
        heading.className = 'gabriel-filter-title';
        heading.textContent = 'Numeric Ranges';
        group.appendChild(heading);
        const grid = document.createElement('div');
        grid.className = 'gabriel-numeric-grid';
        filters.forEach(filter => {
            if (typeof filter.min !== 'number' || typeof filter.max !== 'number') {
                return;
            }
            numericDefaults[filter.column] = { min: filter.min, max: filter.max };
            state.numeric[filter.column] = { min: filter.min, max: filter.max, active: false };
            const control = document.createElement('div');
            control.className = 'gabriel-numeric-control';
            control.dataset.numeric = filter.column;
            const label = document.createElement('div');
            label.className = 'gabriel-numeric-label';
            label.textContent = filter.label;
            control.appendChild(label);
            const sliderWrap = document.createElement('div');
            sliderWrap.className = 'gabriel-numeric-slider';
            const minInput = document.createElement('input');
            minInput.type = 'range';
            minInput.min = String(filter.min);
            minInput.max = String(filter.max);
            minInput.step = String(filter.step);
            minInput.value = String(filter.min);
            const maxInput = document.createElement('input');
            maxInput.type = 'range';
            maxInput.min = String(filter.min);
            maxInput.max = String(filter.max);
            maxInput.step = String(filter.step);
            maxInput.value = String(filter.max);
            sliderWrap.appendChild(minInput);
            sliderWrap.appendChild(maxInput);
            control.appendChild(sliderWrap);
            const values = document.createElement('div');
            values.className = 'gabriel-numeric-values';
            const minValue = document.createElement('span');
            minValue.className = 'gabriel-numeric-value-chip';
            minValue.textContent = formatNumber(filter.min);
            const maxValue = document.createElement('span');
            maxValue.className = 'gabriel-numeric-value-chip';
            maxValue.textContent = formatNumber(filter.max);
            values.appendChild(minValue);
            values.appendChild(maxValue);
            control.appendChild(values);
            grid.appendChild(control);
            numericHandles[filter.column] = { minInput, maxInput, minValue, maxValue };
            const handler = () => handleNumericChange(filter.column);
            minInput.addEventListener('input', handler);
            maxInput.addEventListener('input', handler);
        });
        if (!grid.children.length) {
            return null;
        }
        group.appendChild(grid);
        return group;
    }

    function buildFilters() {
        if (!filtersHost || !filterPanel) {
            return;
        }
        filtersHost.innerHTML = '';
        if (!data.hasFilters) {
            filterPanel.style.display = 'none';
            return;
        }
        filterPanel.style.display = '';
        if (data.categoryFilters && data.categoryFilters.length) {
            const group = createToggleGroup('Categories', data.categoryFilters, 'category');
            filtersHost.appendChild(group);
        }
        if (data.booleanFilters && data.booleanFilters.length) {
            const group = createToggleGroup('Attributes', data.booleanFilters, 'boolean');
            filtersHost.appendChild(group);
        }
        if (data.numericFilters && data.numericFilters.length) {
            const numericGroup = createNumericGroup(data.numericFilters);
            if (numericGroup) {
                filtersHost.appendChild(numericGroup);
            }
        }
    }

    buildFilters();

    if (clearFiltersBtn) {
        clearFiltersBtn.addEventListener('click', () => {
            state.snippets.clear();
            state.bools.clear();
            container.querySelectorAll('.gabriel-filter-toggle[aria-pressed="true"]').forEach(btn => btn.setAttribute('aria-pressed', 'false'));
            Object.entries(numericDefaults).forEach(([column, defaults]) => {
                state.numeric[column] = { min: defaults.min, max: defaults.max, active: false };
                const handle = numericHandles[column];
                if (handle) {
                    handle.minInput.value = String(defaults.min);
                    handle.maxInput.value = String(defaults.max);
                    handle.minValue.textContent = formatNumber(defaults.min);
                    handle.maxValue.textContent = formatNumber(defaults.max);
                }
            });
            applyFilters(true);
        });
    }

    if (prevBtn) {
        prevBtn.addEventListener('click', () => moveIndex(-1));
    }
    if (nextBtn) {
        nextBtn.addEventListener('click', () => moveIndex(1));
    }
    if (randomBtn) {
        randomBtn.addEventListener('click', () => {
            if (!state.active.length) {
                return;
            }
            const idx = Math.floor(Math.random() * state.active.length);
            state.index = idx;
            renderPassage();
            updateStatus();
            updateSlider();
        });
    }
    if (sliderEl) {
        sliderEl.addEventListener('input', event => {
            if (!state.active.length) {
                return;
            }
            const value = parseInt(event.target.value, 10);
            if (Number.isNaN(value)) {
                return;
            }
            state.index = Math.min(state.active.length - 1, Math.max(0, value - 1));
            renderPassage();
            updateStatus();
            updateSlider(false);
        });
    }

    applyFilters(true);
})();
</script>
"""
    )
    viewer_html = viewer_template.substitute(
        viewer_id=viewer_id,
        root_class=root_class,
        slider_max=slider_max,
        slider_count=slider_count,
        note_block=note_block,
        data_json=data_json,
    )
    display(HTML(style_html + viewer_html))


def view(
    df: pd.DataFrame,
    column_name: str,
    attributes: Optional[Union[Mapping[str, Any], Sequence[Any], Any]] = None,
    *,
    header_columns: Optional[Union[Sequence[Any], Any]] = None,
    max_passages: Optional[int] = None,
    font_scale: float = 1.0,
    font_family: Optional[str] = None,
    color_mode: str = "auto",
):
    """View passages and their associated attributes.

    Parameters
    ----------
    df:
        DataFrame containing the passages.
    column_name:
        Column name in ``df`` holding the raw text.
    attributes:
        Attribute columns to render. Accepts sequences of column names, tuples
        of ``(column, label)``, mappings, or the special string
        ``"coded_passages"`` for Codify outputs.
    header_columns:
        Optional sequence of column names (or ``(column, label)`` tuples)
        displayed above each passage. Values are rendered in the provided
        order to expose metadata such as speaker names or timestamps.
    max_passages:
        Optional cap on the number of passages rendered in the notebook.
        When ``None`` (default) all passages are available inside the viewer.
    font_scale:
        Multiplier applied to key font sizes inside the viewer.
    font_family:
        Optional custom font family prepended to the default stack.
    color_mode:
        ``"auto"`` (default), ``"dark"``, or ``"light"`` to force a theme.
    """

    _render_passage_viewer(
        df,
        column_name,
        attributes=attributes,
        header_columns=header_columns,
        max_passages=max_passages,
        font_scale=font_scale,
        font_family=font_family,
        color_mode=color_mode,
    )
    return None


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Sample data
    sample_data = {
        'id': [1, 2, 3],
        'text': [
            "This is a great example of positive text. I really appreciate your help with this matter.",
            "I can't believe how terrible this service is. This is absolutely unacceptable behavior.",
            "Could you please explain how this works? I'm genuinely curious about the process."
        ],
        'positive_sentiment': [
            ["This is a great example of positive text", "I really appreciate your help"],
            [],
            ["I'm genuinely curious about the process"]
        ],
        'negative_sentiment': [
            [],
            ["I can't believe how terrible this service is", "This is absolutely unacceptable behavior"],
            []
        ],
        'questions': [
            [],
            [],
            ["Could you please explain how this works?"]
        ]
    }
    
    df = pd.DataFrame(sample_data)
    categories = ['positive_sentiment', 'negative_sentiment', 'questions']
    
    view(df, 'text', attributes=categories)
