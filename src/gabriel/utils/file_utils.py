from __future__ import annotations

import os
from typing import Iterable, Optional, Any

import pandas as pd

from .logging import get_logger

logger = get_logger(__name__)


def load_files(
    folder_path: str,
    extensions: Optional[Iterable[str]] = None,
    *,
    tag_dict: Optional[dict[str, Any]] = None,
    save_name: str = "gabriel_aggregated_content.csv",
    reset_files: bool = False,
    modality: str = "text",
) -> pd.DataFrame:
    """Aggregate text files from a folder into a single CSV.

    Parameters
    ----------
    folder_path:
        Path to a directory containing text files or to a single file.
    extensions:
        Optional iterable of file extensions (without leading dots) to include.
        When ``None`` all text files are processed.
    tag_dict:
        Optional mapping of substrings to tag values. The first matching
        substring found in a file name determines the ``tag`` column value.
    save_name:
        Name of the output CSV written inside ``folder_path`` (or its parent
        directory when ``folder_path`` points to a file). Defaults to
        ``"gabriel_aggregated_content.csv"``.
    reset_files:
        When ``False`` (default), an existing file at the save location causes a
        :class:`FileExistsError`. Set to ``True`` to overwrite the file.

    modality:
        One of ``"text"``, ``"image"``, or ``"audio"``. Non‑text modalities
        collect file paths rather than file contents.

    Returns
    -------
    DataFrame
        The aggregated contents or file paths of the processed files.
    """

    folder_path = os.path.expanduser(os.path.expandvars(folder_path))
    modality = modality.lower()
    if modality not in {"text", "image", "audio"}:
        raise ValueError("modality must be 'text', 'image', or 'audio'")

    path_key = "path" if modality == "text" else f"{modality}_path"
    if os.path.isfile(folder_path):
        base_dir = os.path.dirname(folder_path)
    else:
        base_dir = folder_path
    save_path = os.path.join(base_dir, save_name)

    if os.path.exists(save_path) and not reset_files:
        raise FileExistsError(
            f"{save_path} exists. Set reset_files=True to overwrite or choose a different save_name."
        )

    extset = {e.lower().lstrip(".") for e in extensions} if extensions else None
    rows: list[dict[str, Any]] = []
    max_layers = 0

    if os.path.isfile(folder_path):
        ext = os.path.splitext(folder_path)[1].lower()
        if modality == "text" and ext in {".csv", ".xlsx", ".xls"}:
            logger.info("Input path is a %s file; saving it directly.", ext)
            if ext == ".csv":
                df = pd.read_csv(folder_path)
            else:
                df = pd.read_excel(folder_path)
            df.to_csv(save_path, index=False)
            print(df.head())
            print(f"Saved aggregated file to {save_path}")
            return df
        else:
            name = os.path.basename(folder_path)
            tag = None
            if tag_dict:
                lower_name = name.lower()
                for key, val in tag_dict.items():
                    if key.lower() in lower_name:
                        tag = val
                        break
            row = {
                "name": name,
                path_key: folder_path,
                "tag": tag,
            }
            if modality == "text":
                with open(folder_path, "r", encoding="utf-8", errors="ignore") as fh:
                    row["content"] = fh.read()
            rows.append(row)
    else:
        for root, _, files in os.walk(folder_path):
            for fname in files:
                if fname == save_name:
                    continue
                ext = os.path.splitext(fname)[1].lower().lstrip(".")
                if extset and ext not in extset:
                    continue
                file_path = os.path.join(root, fname)
                rel = os.path.relpath(file_path, folder_path)
                parts = rel.split(os.sep)
                name = parts[-1]
                layers = parts[:-1]
                max_layers = max(max_layers, len(layers))
                tag = None
                if tag_dict:
                    lower_name = name.lower()
                    for key, val in tag_dict.items():
                        if key.lower() in lower_name:
                            tag = val
                            break
                row: dict[str, Any] = {
                    "name": name,
                    path_key: file_path,
                    "tag": tag,
                }
                if modality == "text":
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as fh:
                        row["content"] = fh.read()
                for i, layer in enumerate(layers, start=1):
                    row[f"layer_{i}"] = layer
                rows.append(row)

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
    if modality == "text":
        cols.append("content")
    else:
        df.drop(columns=["content"], inplace=True, errors="ignore")
    if not df.empty:
        df = df[cols]
    df.to_csv(save_path, index=False)
    print(df.head())
    print(f"Saved aggregated file to {save_path}")
    return df
