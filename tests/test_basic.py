import asyncio
import pandas as pd
import numpy as np
import openai
import pytest

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils import openai_utils, safest_json
from gabriel.tasks.rate import Rate, RateConfig
from gabriel.tasks.deidentify import Deidentifier, DeidentifyConfig
from gabriel.tasks.classify import Classify, ClassifyConfig, _collect_predictions
from gabriel.tasks.extract import Extract, ExtractConfig
from gabriel.tasks.rank import Rank, RankConfig
import gabriel


def test_prompt_template():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    text = tmpl.render(attributes=["a"], descriptions=["desc"], passage="x", object_category="obj", attribute_category="att", format="json")
    assert "desc" in text


def test_ratings_default_scale_prompt():
    tmpl = PromptTemplate.from_package("ratings_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes=["clarity"], scale=None)
    assert "Use integers 0-100" in rendered


def test_shuffled_dict_rendering():
    tmpl = PromptTemplate.from_package("classification_prompt.jinja2")
    rendered = tmpl.render(text="x", attributes={"clarity": "Is the text clear?"})
    assert "OrderedDict" not in rendered
    assert "{" in rendered and "}" in rendered


def test_get_response_dummy():
    responses, _ = asyncio.run(openai_utils.get_response("hi", use_dummy=True))
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_images_dummy():
    responses, _ = asyncio.run(
        openai_utils.get_response("hi", images=["abcd"], use_dummy=True)
    )
    assert responses and responses[0].startswith("DUMMY")


def test_get_response_audio_dummy():
    responses, _ = asyncio.run(
        openai_utils.get_response(
            "hi", audio=[{"data": "abcd", "format": "mp3"}], use_dummy=True
        )
    )
    assert responses and responses[0].startswith("DUMMY")


def test_gpt_audio_modalities(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    openai_utils._clients_async.clear()

    class DummyClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        async def create(self, **kwargs):
            DummyClient.captured = kwargs

            class Msg:
                content = ""

            class Choice:
                message = Msg()

            class Resp:
                choices = [Choice()]

            return Resp()

    dummy = DummyClient()
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_: dummy)

    asyncio.run(
        openai_utils.get_response(
            "hi",
            model="gpt-audio",
            audio=[{"data": "abcd", "format": "mp3"}],
            use_dummy=False,
        )
    )
    assert DummyClient.captured["modalities"] == ["text"]


def test_custom_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    openai_utils._clients_async.clear()
    client = openai_utils._get_client("https://example.com/v1")
    assert str(client.base_url) == "https://example.com/v1/"
    openai_utils._clients_async.clear()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.org/v1")
    client2 = openai_utils._get_client()
    assert str(client2.base_url) == "https://example.org/v1/"


def test_get_embedding_dummy():
    emb, _ = asyncio.run(openai_utils.get_embedding("hi", use_dummy=True))
    assert isinstance(emb, list) and emb and isinstance(emb[0], float)


def test_safest_json_codeblock_list():
    raw = ["```json\n{\n \"speech\": true,\n \"music\": false\n}\n```"]
    parsed = asyncio.run(safest_json(raw))
    assert parsed == {"speech": True, "music": False}


def test_safest_json_invalid_without_fallback():
    parsed = asyncio.run(safest_json("not json"))
    assert parsed is None


def test_gpt5_temperature_warning(caplog):
    """Ensure gpt-5 models ignore temperature and log a warning."""
    with caplog.at_level("WARNING"):
        params = openai_utils._build_params(
            model="gpt-5-mini",
            input_data=[{"role": "user", "content": "hi"}],
            max_output_tokens=None,
            system_instruction="test",
            temperature=0.2,
            tools=None,
            tool_choice=None,
            web_search=False,
            search_context_size="medium",
            json_mode=False,
            expected_schema=None,
            reasoning_effort="medium",
        )
    assert "temperature" not in params
    assert any("does not support temperature" in r.message for r in caplog.records)


def test_get_all_responses_dummy(tmp_path):
    df = asyncio.run(openai_utils.get_all_responses(
        prompts=["a", "b"],
        identifiers=["1", "2"],
        save_path=str(tmp_path / "out.csv"),
        use_dummy=True,
    ))
    assert len(df) == 2
    assert set(["Successful", "Error Log"]).issubset(df.columns)
    assert df["Successful"].all()


def test_get_all_responses_images_dummy(tmp_path):
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            prompt_images={"1": ["abcd"]},
            save_path=str(tmp_path / "img.csv"),
            use_dummy=True,
        )
    )
    assert len(df) == 1


def test_get_all_responses_audio_dummy(tmp_path):
    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["a"],
            identifiers=["1"],
            prompt_audio={"1": [{"data": "abcd", "format": "mp3"}]},
            save_path=str(tmp_path / "aud.csv"),
            use_dummy=True,
        )
    )
    assert len(df) == 1


def test_get_all_responses_custom_callable(tmp_path):
    calls = []

    async def custom(prompt: str, *, n: int) -> list:
        calls.append((prompt, n))
        return [f"CUSTOM::{prompt}"]

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["x", "y"],
            identifiers=["1", "2"],
            save_path=str(tmp_path / "custom.csv"),
            response_fn=custom,
            reset_files=True,
        )
    )
    assert sorted(calls) == [("x", 1), ("y", 1)]
    df = df.sort_values("Identifier").reset_index(drop=True)
    assert df.loc[0, "Response"] == ["CUSTOM::x"]
    assert df.loc[1, "Response"] == ["CUSTOM::y"]
    assert df["Successful"].all()
    assert df["Time Taken"].isna().all()


def test_get_all_responses_custom_usage(tmp_path):
    recorded_kwargs = []

    async def custom(prompt: str, **kwargs) -> tuple:
        recorded_kwargs.append(kwargs)
        return (
            [f"ANS:{prompt}"],
            0.5,
            [
                {
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 7,
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [{"text": f"summary-{prompt}"}],
                        }
                    ],
                }
            ],
        )

    df = asyncio.run(
        openai_utils.get_all_responses(
            prompts=["z"],
            identifiers=["id"],
            save_path=str(tmp_path / "custom_usage.csv"),
            response_fn=custom,
            reasoning_summary="short",
            reset_files=True,
        )
    )
    assert recorded_kwargs and recorded_kwargs[0].get("return_raw") is True
    row = df.iloc[0]
    assert row["Response"] == ["ANS:z"]
    assert row["Input Tokens"] == 10
    assert row["Output Tokens"] == 7
    assert row["Reasoning Tokens"] == 2
    assert row["Reasoning Summary"] == "summary-z"
    assert bool(row["Successful"])
    assert row["Time Taken"] == pytest.approx(0.5)


def test_get_all_embeddings_dummy(tmp_path):
    res = asyncio.run(
        openai_utils.get_all_embeddings(
            texts=["a", "b"],
            identifiers=["1", "2"],
            save_path=str(tmp_path / "emb.pkl"),
            use_dummy=True,
        )
    )
    assert set(res.keys()) == {"1", "2"}


def test_ratings_dummy(tmp_path):
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert not df.empty
    assert "helpfulness" in df.columns


def test_ratings_multirun(tmp_path):
    cfg = RateConfig(attributes={"helpfulness": ""}, save_dir=str(tmp_path), file_name="ratings.csv", use_dummy=True, n_runs=2)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "helpfulness" in df.columns
    disagg = pd.read_csv(tmp_path / "ratings_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"id", "run"}


def test_ratings_ignore_stale_ids(tmp_path):
    """Ensure stale identifiers in existing files are ignored."""
    cfg = RateConfig(
        attributes={"helpfulness": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
    )
    # Pre-create a raw responses file with an unrelated identifier
    raw_path = tmp_path / "ratings_raw_responses.csv"
    stale = pd.DataFrame(
        [
            {
                "Identifier": "stale_batch0",
                "Response": openai_utils._ser(["{\"helpfulness\": 1}"]),
                "Time Taken": 0.1,
                "Input Tokens": 1,
                "Reasoning Tokens": 0,
                "Output Tokens": 1,
                "Reasoning Effort": None,
                "Successful": True,
                "Error Log": openai_utils._ser(None),
            }
        ]
    )
    stale.to_csv(raw_path, index=False)
    task = Rate(cfg)
    data = pd.DataFrame({"text": ["hello"]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "helpfulness" in df.columns


def test_ratings_audio_dummy(tmp_path):
    cfg = RateConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
        modality="audio",
    )
    task = Rate(cfg)
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"abcd")
    data = pd.DataFrame({"audio": [str(audio_path)]})
    df = asyncio.run(task.run(data, column_name="audio"))
    assert "clarity" in df.columns


def test_ratings_image_dummy(tmp_path):
    cfg = RateConfig(
        attributes={"clarity": ""},
        save_dir=str(tmp_path),
        file_name="ratings.csv",
        use_dummy=True,
        modality="image",
    )
    task = Rate(cfg)
    img_path = tmp_path / "test.png"
    img_path.write_bytes(b"abcd")
    data = pd.DataFrame({"image": [str(img_path)]})
    df = asyncio.run(task.run(data, column_name="image"))
    assert "clarity" in df.columns


def test_rank_audio_dummy(tmp_path):
    cfg = RankConfig(
        attributes={"clear": "", "inspiring": ""},
        save_dir=str(tmp_path),
        file_name="rankings.csv",
        use_dummy=True,
        modality="audio",
        n_rounds=1,
        matches_per_round=1,
        n_parallels=5,
    )
    task = Rank(cfg)
    # Provide pre-encoded audio so no actual files are needed
    data = pd.DataFrame(
        {
            "audio": [
                [{"data": "abcd", "format": "mp3"}],
                [{"data": "efgh", "format": "mp3"}],
            ]
        }
    )
    df = asyncio.run(task.run(data, column_name="audio"))
    assert "clear" in df.columns and "inspiring" in df.columns


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_dir=str(tmp_path), file_name="deid.csv", use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, column_name="text"))
    assert "deidentified_text" in df.columns


def test_classification_dummy(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a", "b"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "yes" in res.columns
    assert "predicted_classes" in res.columns
    assert res.predicted_classes.tolist() == [[], []]


def test_extraction_dummy(tmp_path):
    cfg = ExtractConfig(attributes={"year": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Extract(cfg)
    df = pd.DataFrame({"txt": ["a"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "year" in res.columns


def test_classification_multirun(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True, n_runs=2)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "yes" in res.columns
    assert res.predicted_classes.iloc[0] == []
    disagg = pd.read_csv(tmp_path / "classify_responses_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"text", "run"}


def test_collect_predictions_np_bool():
    row = pd.Series({"speech": np.bool_(True), "beeps": np.bool_(False), "space": None})
    assert _collect_predictions(row) == ["speech"]


def test_classify_parse_dict(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    parsed = asyncio.run(task._parse({"yes": True}, ["yes"]))
    assert parsed["yes"] is True
    

def test_api_wrappers(tmp_path):
    df = pd.DataFrame({"txt": ["hello"]})
    rated = asyncio.run(
        gabriel.rate(
            df,
            "txt",
            attributes={"clarity": ""},
            save_dir=str(tmp_path / "rate"),
            use_dummy=True,
        )
    )
    assert "clarity" in rated.columns

    classified = asyncio.run(
        gabriel.classify(
            df,
            "txt",
            labels={"yes": ""},
            save_dir=str(tmp_path / "cls"),
            use_dummy=True,
        )
    )
    assert "yes" in classified.columns

    extracted = asyncio.run(
        gabriel.extract(
            df,
            "txt",
            attributes={"year": ""},
            save_dir=str(tmp_path / "extr"),
            use_dummy=True,
        )
    )
    assert "year" in extracted.columns

    deidentified = asyncio.run(
        gabriel.deidentify(
            df,
            "txt",
            save_dir=str(tmp_path / "deid"),
            use_dummy=True,
        )
    )
    assert "deidentified_text" in deidentified.columns

    custom = asyncio.run(
        gabriel.whatever(
            prompts=["hello"],
            identifiers=["1"],
            save_dir=str(tmp_path / "cust"),
            file_name="out.csv",
            use_dummy=True,
        )
    )
    assert len(custom) == 1


def test_paraphrase_api(tmp_path):
    data = pd.DataFrame({"txt": ["hello"]})
    df = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para"),
            use_dummy=True,
        )
    )
    assert "txt_revised" in df.columns and len(df) == 1
    df_multi = asyncio.run(
        gabriel.paraphrase(
            data,
            "txt",
            instructions="reword",
            save_dir=str(tmp_path / "para_multi"),
            use_dummy=True,
            n_revisions=2,
        )
    )
    assert "txt_revised_1" in df_multi.columns and "txt_revised_2" in df_multi.columns

