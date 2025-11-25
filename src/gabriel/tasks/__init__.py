"""Task implementations for GABRIEL."""

from importlib import import_module
from typing import List

_lazy_imports = {
    "Rate": ".rate",
    "RateConfig": ".rate",
    "Deidentifier": ".deidentify",
    "DeidentifyConfig": ".deidentify",
    "EloRater": ".elo",
    "EloConfig": ".elo",
    "Classify": ".classify",
    "ClassifyConfig": ".classify",
    "Rank": ".rank",
    "RankConfig": ".rank",
    "Codify": ".codify",
    "CodifyConfig": ".codify",
    "Paraphrase": ".paraphrase",
    "ParaphraseConfig": ".paraphrase",
    "Extract": ".extract",
    "ExtractConfig": ".extract",
    "Regional": ".regional",
    "RegionalConfig": ".regional",
    "CountyCounter": ".county_counter",
    "Compare": ".compare",
    "CompareConfig": ".compare",
    "Merge": ".merge",
    "MergeConfig": ".merge",
    "Deduplicate": ".deduplicate",
    "DeduplicateConfig": ".deduplicate",
    "Bucket": ".bucket",
    "BucketConfig": ".bucket",
    "Discover": ".discover",
    "DiscoverConfig": ".discover",
    "Seed": ".seed",
    "SeedConfig": ".seed",
    "Filter": ".filter",
    "FilterConfig": ".filter",
    "Whatever": ".whatever",
    "WhateverConfig": ".whatever",
    "Ideate": ".ideate",
    "IdeateConfig": ".ideate",
    "DebiasPipeline": ".debias",
    "DebiasConfig": ".debias",
    "DebiasResult": ".debias",
    "DebiasRegressionResult": ".debias",
    "MeasurementMode": ".debias",
    "RemovalMethod": ".debias",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name: str):
    if name in _lazy_imports:
        module = import_module(_lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(name)


def __dir__() -> List[str]:
    return __all__
