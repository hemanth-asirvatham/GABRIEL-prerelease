import asyncio
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from gabriel.tasks.debias import DebiasConfig, DebiasPipeline
from gabriel.tasks.codify import Codify
from gabriel.tasks.paraphrase import Paraphrase


def _range_index(n: int) -> pd.Index:
    return pd.Index(range(n), name="__debias_row_id")


def test_prepare_codify_variants_strips_text(monkeypatch, tmp_path):
    df = pd.DataFrame({"text": ["alpha BIAS0 omega", "beta BIAS1 gamma"]})
    cfg = DebiasConfig(
        mode="rate",
        measurement_attribute="bias_score",
        attributes={"bias_score": "desc"},
        signal_dictionary={"bias_score": "remove"},
        removal_method="codify",
        save_dir=str(tmp_path),
        strip_percentages=[100],
        run_name="unit_codify",
        verbose=False,
    )
    pipeline = DebiasPipeline(cfg)

    async def fake_codify_run(
        self,
        df_in: pd.DataFrame,
        column_name: str,
        *,
        categories=None,
        additional_instructions="",
        reset_files=False,
        **kwargs,
    ) -> pd.DataFrame:
        assert categories == cfg.signal_dictionary
        assert additional_instructions == ""
        idx = df_in["__debias_row_id"].astype(int).tolist()
        snippets = [[f"BIAS{i}"] for i in idx]
        out = pd.DataFrame({
            column_name: df_in[column_name].tolist(),
            cfg.removal_attribute: snippets,
        })
        out.index = pd.Index(idx, name="__debias_row_id")
        return out

    monkeypatch.setattr(Codify, "run", fake_codify_run)

    df_local = df.copy()
    df_local.index = _range_index(len(df_local))
    variants = asyncio.run(pipeline._prepare_codify_variants(df_local, "text"))

    assert "stripped_100pct" in variants
    variant_col = variants["stripped_100pct"]["text_column"]
    assert list(df_local[variant_col]) == ["alpha omega", "beta gamma"]


def test_prepare_paraphrase_variant_builds_instructions(monkeypatch, tmp_path):
    df = pd.DataFrame({"text": ["alpha BIAS omega", "beta BIAS gamma"]})
    cfg = DebiasConfig(
        mode="rate",
        measurement_attribute="bias_score",
        attributes={"bias_score": "desc"},
        signal_dictionary={"bias_score": "Mentions of BIAS."},
        removal_method="paraphrase",
        save_dir=str(tmp_path),
        run_name="unit_paraphrase",
        verbose=False,
    )
    pipeline = DebiasPipeline(cfg)
    captured: dict[str, str] = {}

    async def fake_paraphrase_run(
        self,
        df_in: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        idx = df_in["__debias_row_id"].astype(int).tolist()
        revised_name = self.cfg.revised_column_name or f"{column_name} (revised)"
        revised = [str(text).replace("BIAS", "").strip() for text in df_in[column_name]]
        out = pd.DataFrame({revised_name: revised})
        out.index = pd.Index(idx, name="__debias_row_id")
        captured["instructions"] = self.cfg.instructions
        return out

    monkeypatch.setattr(Paraphrase, "run", fake_paraphrase_run)

    df_local = df.copy()
    df_local.index = _range_index(len(df_local))
    variants = asyncio.run(pipeline._prepare_paraphrase_variant(df_local, "text"))

    assert "paraphrase" in variants
    variant_col = variants["paraphrase"]["text_column"]
    cleaned = [" ".join(val.split()) for val in df_local[variant_col]]
    assert cleaned == ["alpha omega", "beta gamma"]
    assert "bias_score" in captured["instructions"]


def test_debias_config_defaults_emit_notice(capsys, tmp_path):
    cfg = DebiasConfig(
        mode="rate",
        attributes={"bias_score": "desc"},
        signal_dictionary={"bias_flag": "Mentions of the token FLAG."},
        removal_method="codify",
        save_dir=str(tmp_path),
        strip_percentages=[100],
        run_name="default_notice",
        verbose=True,
    )
    DebiasPipeline(cfg)
    output = capsys.readouterr().out
    assert "measurement_attribute not provided" in output
    assert "removal_attribute not provided" in output


def test_debias_pipeline_codify_flow(monkeypatch, tmp_path):
    texts = [
        "Row0 BIAS narrative",
        "Row1 BIAS commentary",
        "Row2 BIAS discussion",
        "Row3 BIAS overview",
    ]
    df = pd.DataFrame({"text": texts})
    cfg = DebiasConfig(
        mode="rate",
        measurement_attribute="bias_score",
        attributes={
            "bias_score": "Intensity of bias language",
            "context_score": "Contextual richness",
        },
        signal_dictionary={"bias_score": "Mentions of the token BIAS."},
        removal_method="codify",
        save_dir=str(tmp_path),
        strip_percentages=[100],
        run_name="codify_pipeline",
        verbose=False,
    )
    pipeline = DebiasPipeline(cfg)

    async def fake_codify_run(
        self,
        df_in: pd.DataFrame,
        column_name: str,
        *,
        categories=None,
        additional_instructions="",
        reset_files=False,
        **kwargs,
    ) -> pd.DataFrame:
        assert categories == cfg.signal_dictionary
        snippets = [["BIAS"]] * len(df_in)
        out = pd.DataFrame({
            column_name: df_in[column_name].tolist(),
            cfg.removal_attribute: snippets,
        })
        out.index = pd.Index(
            df_in["__debias_row_id"].astype(int).tolist(), name="__debias_row_id"
        )
        return out

    monkeypatch.setattr(Codify, "run", fake_codify_run)

    variant_col = f"text ({cfg.removal_attribute} stripped 100%)"
    index = _range_index(len(df))
    measurement_map = {
        ("rate", "original", "text"): pd.DataFrame(
            {
                "bias_score": [0.0, 5.0, 10.0, 20.0],
                "context_score": [1.0, 2.0, 1.5, 0.5],
            },
            index=index,
        ),
        ("rate", "stripped_100pct", variant_col): pd.DataFrame(
            {
                "bias_score": [0.0, 2.0, 4.0, 5.0],
                "context_score": [0.5, 1.5, 1.0, 0.25],
            },
            index=index,
        ),
    }

    async def fake_run_measurement(
        self,
        df_in: pd.DataFrame,
        *,
        column_name: str,
        mode: str,
        save_label: str,
        attributes,
        template_path,
        extra_kwargs,
        default_model,
    ) -> pd.DataFrame:
        key = (mode, save_label, column_name)
        assert key in measurement_map, key
        return measurement_map[key].copy()

    monkeypatch.setattr(DebiasPipeline, "_run_measurement", fake_run_measurement)

    result = asyncio.run(pipeline.run(df, "text"))

    results_df = result.results
    assert variant_col in results_df.columns
    assert all("BIAS" not in val for val in results_df[variant_col])

    residual_col = "bias_score__residual_stripped_100pct"
    debiased_col = "bias_score__debiased_stripped_100pct"
    pdt.assert_series_equal(
        results_df[residual_col],
        results_df[debiased_col],
        check_names=False,
    )

    regression = result.regression["stripped_100pct"]
    assert regression.strip_percentage == 100
    assert regression.regression is not None
    assert pytest.approx(regression.mean_original) == results_df["bias_score"].mean()

    metadata = result.metadata
    assert metadata["config"]["removal_method"] == "codify"
    assert metadata["config"]["model"] == "gpt-5-mini"
    metadata_path = Path(metadata["metadata_path"])
    assert metadata_path.exists()
    saved_df = pd.read_csv(metadata["result_path"])
    pdt.assert_frame_equal(saved_df, results_df, check_dtype=False)


def test_debias_pipeline_paraphrase_flow(monkeypatch, tmp_path):
    texts = [
        "County BIAS overview",
        "City BIAS snapshot",
        "Region BIAS digest",
        "Village BIAS memo",
    ]
    df = pd.DataFrame({"text": texts})
    cfg = DebiasConfig(
        mode="rate",
        measurement_attribute="bias_score",
        attributes={
            "bias_score": "Bias intensity",
            "context_score": "Context attribute",
        },
        signal_dictionary={"bias_score": "References to the token BIAS."},
        removal_method="paraphrase",
        save_dir=str(tmp_path),
        run_name="paraphrase_pipeline",
        verbose=False,
    )
    pipeline = DebiasPipeline(cfg)
    captured = {}

    async def fake_paraphrase_run(
        self,
        df_in: pd.DataFrame,
        column_name: str,
        *,
        reset_files: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        idx = df_in["__debias_row_id"].astype(int).tolist()
        revised_name = self.cfg.revised_column_name or (
            f"{column_name} ({cfg.removal_attribute} stripped paraphrase)"
        )
        revised = [str(text).replace("BIAS", "").strip() for text in df_in[column_name]]
        out = pd.DataFrame({revised_name: revised})
        out.index = pd.Index(
            df_in["__debias_row_id"].astype(int).tolist(), name="__debias_row_id"
        )
        captured["instructions"] = self.cfg.instructions
        return out

    monkeypatch.setattr(Paraphrase, "run", fake_paraphrase_run)

    variant_col = f"text ({cfg.removal_attribute} stripped paraphrase)"
    index = _range_index(len(df))
    measurement_map = {
        ("rate", "original", "text"): pd.DataFrame(
            {
                "bias_score": [1.0, 2.0, 3.0, 4.0],
                "context_score": [0.0, 1.0, 0.5, 0.25],
            },
            index=index,
        ),
        ("rate", "paraphrase", variant_col): pd.DataFrame(
            {
                "bias_score": [0.5, 1.5, 1.0, 2.0],
                "context_score": [0.0, 0.8, 0.4, 0.2],
            },
            index=index,
        ),
    }

    async def fake_run_measurement(
        self,
        df_in: pd.DataFrame,
        *,
        column_name: str,
        mode: str,
        save_label: str,
        attributes,
        template_path,
        extra_kwargs,
        default_model,
    ) -> pd.DataFrame:
        key = (mode, save_label, column_name)
        assert key in measurement_map, key
        return measurement_map[key].copy()

    monkeypatch.setattr(DebiasPipeline, "_run_measurement", fake_run_measurement)

    result = asyncio.run(pipeline.run(df, "text"))

    results_df = result.results
    assert variant_col in results_df.columns
    assert all("BIAS" not in val for val in results_df[variant_col])

    residual_col = "bias_score__residual_paraphrase"
    debiased_col = "bias_score__debiased_paraphrase"
    pdt.assert_series_equal(
        results_df[residual_col],
        results_df[debiased_col],
        check_names=False,
    )

    regression = result.regression["paraphrase"]
    assert regression.strip_percentage is None
    assert regression.regression is not None
    assert pytest.approx(regression.mean_stripped) == results_df[
        "bias_score (bias_score stripped (paraphrase))"
    ].mean()

    metadata = result.metadata
    assert metadata["config"]["removal_method"] == "paraphrase"
    saved_df = pd.read_csv(metadata["result_path"])
    pdt.assert_frame_equal(saved_df, results_df, check_dtype=False)
    assert "bias_score" in captured["instructions"]


def test_debias_pipeline_supports_distinct_attributes(monkeypatch, tmp_path):
    texts = [
        "Row0 FLAG narrative",
        "Row1 FLAG commentary",
        "Row2 FLAG discussion",
        "Row3 FLAG overview",
    ]
    df = pd.DataFrame({"text": texts})
    cfg = DebiasConfig(
        mode="rate",
        attributes={"bias_score": "Intensity of bias language"},
        signal_dictionary={
            "bias_flag": "Mentions of the token FLAG.",
            "alternate_flag": "Secondary flag to leave untouched.",
        },
        removal_method="codify",
        save_dir=str(tmp_path),
        strip_percentages=[100],
        run_name="distinct_attrs_pipeline",
        verbose=False,
    )
    pipeline = DebiasPipeline(cfg)

    assert cfg.measurement_attribute == "bias_score"
    assert cfg.removal_attribute == "bias_flag"
    assert cfg.categories_to_strip == ["bias_flag"]

    async def fake_codify_run(
        self,
        df_in: pd.DataFrame,
        column_name: str,
        *,
        categories=None,
        additional_instructions="",
        reset_files=False,
        **kwargs,
    ) -> pd.DataFrame:
        assert categories == cfg.signal_dictionary
        idx = df_in["__debias_row_id"].astype(int).tolist()
        snippets = [["FLAG"]] * len(df_in)
        out = pd.DataFrame(
            {
                column_name: df_in[column_name].tolist(),
                cfg.removal_attribute: snippets,
            }
        )
        out.index = pd.Index(idx, name="__debias_row_id")
        return out

    monkeypatch.setattr(Codify, "run", fake_codify_run)

    variant_col = f"text ({cfg.removal_attribute} stripped 100%)"
    index = _range_index(len(df))
    measurement_map = {
        ("rate", "original", "text"): pd.DataFrame(
            {
                "bias_score": [0.0, 5.0, 10.0, 20.0],
            },
            index=index,
        ),
        ("rate", "stripped_100pct", variant_col): pd.DataFrame(
            {
                "bias_score": [0.0, 1.0, 2.0, 3.0],
            },
            index=index,
        ),
    }

    async def fake_run_measurement(
        self,
        df_in: pd.DataFrame,
        *,
        column_name: str,
        mode: str,
        save_label: str,
        attributes,
        template_path,
        extra_kwargs,
        default_model,
    ) -> pd.DataFrame:
        key = (mode, save_label, column_name)
        assert key in measurement_map, key
        return measurement_map[key].copy()

    monkeypatch.setattr(DebiasPipeline, "_run_measurement", fake_run_measurement)

    result = asyncio.run(pipeline.run(df, "text"))

    results_df = result.results
    assert variant_col in results_df.columns
    measurement_variant = (
        f"{cfg.measurement_attribute} ({cfg.removal_attribute} stripped 100%)"
    )
    assert measurement_variant in results_df.columns
    residual_col = "bias_score__residual_stripped_100pct"
    debiased_col = "bias_score__debiased_stripped_100pct"
    pdt.assert_series_equal(
        results_df[residual_col],
        results_df[debiased_col],
        check_names=False,
    )
    assert result.metadata["config"]["removal_attribute"] == "bias_flag"
    assert result.metadata["config"]["measurement_attribute"] == "bias_score"
