import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from gabriel.utils.plot_utils import bar_plot


@pytest.fixture(autouse=True)
def close_plots():
    try:
        yield
    finally:
        plt.close("all")


def _extract_bar_values():
    fig_numbers = plt.get_fignums()
    assert fig_numbers, "bar_plot should create at least one figure"
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.containers, "plot should contain bar containers"
    container = ax.containers[0]
    return list(container.datavalues)


def test_bar_plot_counts_defaults_to_frequencies():
    df = pd.DataFrame(
        {
            "fruit": [
                "apple",
                "banana",
                "apple",
                "apple",
                "carrot",
                "banana",
                "banana",
            ]
        }
    )

    bar_plot(data=df, category_column="fruit")

    observed = _extract_bar_values()
    expected = list(df["fruit"].astype(str).value_counts())
    assert observed == expected


def test_bar_plot_counts_as_percentages():
    df = pd.DataFrame(
        {
            "pet": [
                "cat",
                "dog",
                "cat",
                "cat",
                "parrot",
                "dog",
            ]
        }
    )

    bar_plot(data=df, category_column="pet", as_percent=True)

    observed = _extract_bar_values()
    counts = df["pet"].astype(str).value_counts()
    expected = list((counts / counts.sum()) * 100)
    assert pytest.approx(observed, rel=1e-6) == expected


def test_bar_plot_counts_respects_category_cap():
    df = pd.DataFrame({"label": [f"item_{idx % 20}" for idx in range(60)]})

    bar_plot(data=df, category_column="label", category_cap=5)

    observed = _extract_bar_values()
    counts = df["label"].astype(str).value_counts().head(5)
    assert len(observed) == 5
    assert observed == list(counts)
