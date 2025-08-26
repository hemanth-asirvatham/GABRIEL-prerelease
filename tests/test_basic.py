import asyncio
import pandas as pd

from gabriel.core.prompt_template import PromptTemplate
from gabriel.utils import openai_utils, safest_json
from gabriel.tasks.rate import Rate, RateConfig
from gabriel.tasks.deidentify import Deidentifier, DeidentifyConfig
from gabriel.tasks.classify import Classify, ClassifyConfig
from gabriel.tasks.extract import Extract, ExtractConfig
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


def test_deidentifier_dummy(tmp_path):
    cfg = DeidentifyConfig(save_dir=str(tmp_path), file_name="deid.csv", use_dummy=True)
    task = Deidentifier(cfg)
    data = pd.DataFrame({"text": ["John went to Paris."]})
    df = asyncio.run(task.run(data, text_column="text"))
    assert "deidentified_text" in df.columns


def test_classification_dummy(tmp_path):
    cfg = ClassifyConfig(labels={"yes": ""}, save_dir=str(tmp_path), use_dummy=True)
    task = Classify(cfg)
    df = pd.DataFrame({"txt": ["a", "b"]})
    res = asyncio.run(task.run(df, column_name="txt"))
    assert "yes" in res.columns


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
    disagg = pd.read_csv(tmp_path / "classify_responses_full_disaggregated.csv", index_col=[0, 1])
    assert set(disagg.index.names) == {"text", "run"}


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

