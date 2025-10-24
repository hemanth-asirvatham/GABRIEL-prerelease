import zipfile

import pandas as pd

from gabriel.tasks.discover import Discover, DiscoverConfig


def test_discover_exports_zip(tmp_path):
    cfg = DiscoverConfig(save_dir=str(tmp_path))
    disc = Discover(cfg)
    result = {
        "candidates": pd.DataFrame({"term": ["alpha"], "definition": ["first"]}),
        "buckets": {"alpha": "first bucket"},
        "classification": pd.DataFrame({"alpha_actual": [True], "alpha_inverted": [False]}),
        "summary": pd.DataFrame({"label": ["alpha"], "actual_true": [1]}),
    }
    disc._export_result_archive(result)
    archive_path = tmp_path / "discover_results_export.zip"
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as zf:
        names = set(zf.namelist())
        assert "candidates.csv" in names
        assert "classification.csv" in names
        assert "buckets.csv" in names
        bucket_data = zf.read("buckets.csv").decode("utf-8")
        assert "first bucket" in bucket_data
