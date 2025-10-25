import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from gabriel.utils.plot_utils import bar_plot, line_plot


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


def test_bar_plot_horizontal_descending_places_largest_at_top():
    df = pd.DataFrame(
        {
            "label": ["alpha", "beta", "gamma", "beta"],
            "value": [2, 5, 1, 3],
        }
    )

    bar_plot(
        data=df,
        category_column="label",
        value_column="value",
        orientation="horizontal",
    )

    fig_numbers = plt.get_fignums()
    assert fig_numbers, "bar_plot should create at least one figure"
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.yaxis_inverted(), "horizontal charts should display largest values at the top"


def test_bar_plot_vertical_auto_size_prefers_wider_layout():
    df = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "d", "e", "f"],
            "score": [1, 2, 3, 4, 5, 6],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score")

    fig_numbers = plt.get_fignums()
    assert fig_numbers, "auto-sized bar plot should create a figure"
    fig = plt.figure(fig_numbers[0])
    assert fig.get_figwidth() >= fig.get_figheight(), "auto sizing should prefer wider-than-tall figures"


def test_bar_plot_long_labels_expand_figure_width():
    long_label = "This is an extremely long feature description that should wrap across multiple lines"
    df = pd.DataFrame(
        {
            "feature": [long_label + str(idx) for idx in range(3)],
            "score": [1.0, 2.0, 3.0],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score")

    fig_numbers = plt.get_fignums()
    assert fig_numbers
    fig = plt.figure(fig_numbers[0])
    assert fig.get_figwidth() > 13.0, "long labels should trigger a wider figure"


def test_bar_plot_aliases_x_label_font_size():
    df = pd.DataFrame(
        {
            "feature": ["alpha", "beta"],
            "score": [1.0, 2.0],
        }
    )

    bar_plot(data=df, category_column="feature", value_column="score", x_label_font_size=7)

    fig_numbers = plt.get_fignums()
    fig = plt.figure(fig_numbers[0])
    fig.canvas.draw()
    ax = fig.axes[0]
    tick_sizes = {round(tick.get_fontsize()) for tick in ax.get_xticklabels() if tick.get_text()}
    assert 7 in tick_sizes


def test_bar_plot_removed_wrap_reference_param_raises():
    df = pd.DataFrame(
        {
            "feature": ["alpha", "beta"],
        }
    )

    with pytest.raises(TypeError, match="wrap_scale_reference has been removed"):
        bar_plot(data=df, category_column="feature", wrap_scale_reference=20)


def test_line_plot_accepts_multiple_y_columns_without_series():
    df = pd.DataFrame(
        {
            "year": [2020, 2020, 2021, 2021],
            "alpha": [1.0, 3.0, 5.0, 7.0],
            "beta": [2.0, 4.0, 6.0, 8.0],
        }
    )

    figs_axes = line_plot(
        df,
        x="year",
        y=["alpha", "beta"],
        show=False,
        gradient=False,
        grid=False,
    )

    assert len(figs_axes) == 1
    fig, ax = figs_axes[0]
    series_data = {line.get_label(): line for line in ax.lines}
    assert set(series_data) >= {"alpha", "beta"}
    alpha_line = series_data["alpha"]
    beta_line = series_data["beta"]
    assert list(alpha_line.get_xdata()) == [2020, 2021]
    assert list(beta_line.get_xdata()) == [2020, 2021]
    assert alpha_line.get_ydata().tolist() == [2.0, 6.0]
    assert beta_line.get_ydata().tolist() == [3.0, 7.0]
