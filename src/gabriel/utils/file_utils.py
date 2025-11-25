from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)

TEXTUAL_MODALITIES = {"text", "entity", "web"}
PATH_MODALITIES = {"image", "audio"}
ALL_MODALITIES = TEXTUAL_MODALITIES | PATH_MODALITIES
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".svg",
}
AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".opus",
    ".aiff",
    ".aif",
    ".aifc",
    ".wma",
    ".alac",
}
IMAGE_EXTENSION_SUFFIXES = {ext.lstrip(".") for ext in IMAGE_EXTENSIONS}
AUDIO_EXTENSION_SUFFIXES = {ext.lstrip(".") for ext in AUDIO_EXTENSIONS}
PATH_MODALITY_SUFFIXES = IMAGE_EXTENSION_SUFFIXES | AUDIO_EXTENSION_SUFFIXES


def load(
    folder_path: str,
    extensions: Optional[Iterable[str]] = None,
    *,
    tag_dict: Optional[Dict[str, Any]] = None,
    save_name: str = "gabriel_aggregated_content.csv",
    save_dir: Optional[str] = None,
    reset_files: bool = False,
    modality: Optional[str] = None,
) -> pd.DataFrame:
    """Aggregate files from a folder into a single CSV.

    Parameters
    ----------
    folder_path:
        Path to a directory containing media files or to a single file. When a
        CSV/Excel file is provided, it is loaded directly without creating a
        copy.
    extensions:
        Optional iterable of file extensions (without leading dots) to include.
        When ``None`` all files are processed.
    tag_dict:
        Optional mapping of substrings to tag values. The first matching
        substring found in a file name determines the ``tag`` column value.
    save_name:
        Name of the output CSV written inside ``save_dir``. Defaults to
        ``"gabriel_aggregated_content.csv"``.
    save_dir:
        Optional directory for the aggregated CSV. When omitted, the data is
        saved inside ``folder_path`` (or the parent directory if
        ``folder_path`` points to a file).
    reset_files:
        When ``False`` (default), an existing file at ``save_path`` is reused
        instead of being regenerated. Set to ``True`` to overwrite the file.
    modality:
        Optional modality hint. ``"text"``, ``"entity"``, and ``"web"`` are
        treated as text; ``"image"`` and ``"audio"`` collect file paths. When
        ``None`` (default) the modality is inferred from the first matching file.

    Returns
    -------
    DataFrame
        The aggregated contents or file paths of the processed files.
    """

    folder_path = os.path.expanduser(os.path.expandvars(folder_path))
    target_dir = _resolve_save_directory(folder_path, save_dir)
    save_path = os.path.join(target_dir, save_name)

    if os.path.exists(save_path) and not reset_files:
        logger.info("Loading existing aggregated file from %s", save_path)
        df = _read_tabular_file(save_path)
        print(df.head())
        print(f"Loaded existing aggregated file from {save_path}")
        return df

    extset = {e.lower().lstrip(".") for e in extensions} if extensions else None
    modality = _resolve_modality(folder_path, extset, save_name, modality)
    is_textual = _is_textual_modality(modality)

    path_key = "path"
    rows: List[Dict[str, Any]] = []
    max_layers = 0

    if os.path.isfile(folder_path):
        ext = os.path.splitext(folder_path)[1].lower()
        if is_textual and ext in TABULAR_EXTENSIONS:
            logger.info(
                "Input path %s is a tabular file; loading it without creating a copy.",
                folder_path,
            )
            df = _read_tabular_file(folder_path)
            print(df.head())
            print(f"Loaded existing file from {folder_path}")
            return df
        name = os.path.basename(folder_path)
        rows.append(
            _build_row(
                file_path=folder_path,
                name=name,
                layers=(),
                tag_dict=tag_dict,
                is_textual=is_textual,
            )
        )
    else:
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname == save_name:
                    continue
                ext = os.path.splitext(fname)[1].lower()
                short_ext = ext.lstrip(".")
                if not _should_include_file(short_ext, modality, extset):
                    continue
                file_path = os.path.join(root, fname)
                rel = os.path.relpath(file_path, folder_path)
                parts = rel.split(os.sep)
                name = parts[-1]
                layers = parts[:-1]
                max_layers = max(max_layers, len(layers))
                rows.append(
                    _build_row(
                        file_path=file_path,
                        name=name,
                        layers=layers,
                        tag_dict=tag_dict,
                        is_textual=is_textual,
                    )
                )

    df = pd.DataFrame(rows)
    for i in range(1, max_layers + 1):
        col = f"layer_{i}"
        if col not in df.columns:
            df[col] = None

    cols = ["name", path_key] + [f"layer_{i}" for i in range(1, max_layers + 1)]
    if tag_dict:
        cols.append("tag")
    else:
        df.drop(columns=["tag"], inplace=True, errors="ignore")
    if is_textual:
        cols.append("text")
    else:
        df.drop(columns=["text"], inplace=True, errors="ignore")
    if not df.empty:
        df = df[cols]
    df.to_csv(save_path, index=False)
    print(df.head())
    print(f"Saved aggregated file to {save_path}")
    return df


def _build_row(
    *,
    file_path: str,
    name: str,
    layers: Iterable[str],
    tag_dict: Optional[Dict[str, Any]],
    is_textual: bool,
) -> Dict[str, Any]:
    tag = _match_tag(name, tag_dict)
    row: Dict[str, Any] = {
        "name": name,
        "path": file_path,
        "tag": tag,
    }
    if is_textual:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
            row["text"] = fh.read()
    for i, layer in enumerate(layers, start=1):
        row[f"layer_{i}"] = layer
    return row


def _match_tag(name: str, tag_dict: Optional[Dict[str, Any]]) -> Optional[Any]:
    if not tag_dict:
        return None
    lower_name = name.lower()
    for key, val in tag_dict.items():
        if key.lower() in lower_name:
            return val
    return None


def _resolve_modality(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
    requested_modality: Optional[str],
) -> str:
    if requested_modality:
        normalized = requested_modality.lower()
        if normalized not in ALL_MODALITIES:
            logger.info(
                "Unknown modality '%s'; defaulting to text-style processing.",
                normalized,
            )
        return normalized
    detected = _detect_modality(folder_path, extset, save_name)
    logger.info("Detected %s modality for %s", detected, folder_path)
    return detected


def _detect_modality(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
) -> str:
    candidate = _find_candidate_file(folder_path, extset, save_name)
    if not candidate:
        return "text"
    ext = os.path.splitext(candidate)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "text"


def _find_candidate_file(
    folder_path: str,
    extset: Optional[Set[str]],
    save_name: str,
) -> Optional[str]:
    if os.path.isfile(folder_path):
        return folder_path
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if fname == save_name:
                continue
            short_ext = os.path.splitext(fname)[1].lower().lstrip(".")
            if extset and short_ext not in extset:
                continue
            return os.path.join(root, fname)
    return None


def _is_textual_modality(modality: str) -> bool:
    if modality in TEXTUAL_MODALITIES:
        return True
    if modality in PATH_MODALITIES:
        return False
    return True


def _should_include_file(
    short_ext: str,
    modality: str,
    extset: Optional[Set[str]],
) -> bool:
    if extset and short_ext not in extset:
        return False
    if modality == "image":
        return short_ext in IMAGE_EXTENSION_SUFFIXES
    if modality == "audio":
        return short_ext in AUDIO_EXTENSION_SUFFIXES
    # Text-style modalities should skip obvious binary media when no explicit filters
    if short_ext in PATH_MODALITY_SUFFIXES:
        return False
    return True


def _resolve_save_directory(folder_path: str, save_dir: Optional[str]) -> str:
    if save_dir:
        resolved = os.path.expanduser(os.path.expandvars(save_dir))
    else:
        if os.path.isdir(folder_path):
            resolved = folder_path
        else:
            parent = os.path.dirname(folder_path)
            if not parent:
                parent = os.path.dirname(os.path.abspath(folder_path))
            resolved = parent
    if not resolved:
        resolved = os.getcwd()
    if os.path.isfile(resolved):
        raise ValueError(f"save_dir must be a directory path, got file {resolved}")
    os.makedirs(resolved, exist_ok=True)
    return resolved


def _read_tabular_file(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)
