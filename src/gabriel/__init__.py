"""GABRIEL: LLM-based social science analysis toolkit."""

from importlib.metadata import PackageNotFoundError, version as _v

from . import tasks as _tasks
from .api import (
    rate,
    classify,
    extract,
    deidentify,
    rank,
    codify,
    paraphrase,
    compare,
    discover,
    deduplicate,
    merge,
    filter,
    debias,
    whatever,
    view_coded_passages,
)
from .utils import load_files

try:
    __version__ = _v("gabriel")
except PackageNotFoundError:  # pragma: no cover - package not installed
    from ._version import __version__

__all__ = list(_tasks.__all__) + [
    "rate",
    "classify",
    "extract",
    "deidentify",
    "rank",
    "codify",
    "paraphrase",
    "compare",
    "discover",
    "deduplicate",
    "merge",
    "filter",
    "debias",
    "whatever",
    "view_coded_passages",
    "load_files",
]


def __getattr__(name: str):
    if name in _tasks.__all__:
        return getattr(_tasks, name)
    raise AttributeError(name)
