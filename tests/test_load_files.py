import os
import pandas as pd
import pytest

from gabriel import load


def test_load_basic(tmp_path):
    base = tmp_path / "data"
    sub = base / "round"
    base.mkdir()
    sub.mkdir()

    (base / "baseline.txt").write_text("hello")
    (sub / "round_1.txt").write_text("world")
    (sub / "round_2.txt").write_text("!")

    tag_dict = {"baseline": 1, "round_1": 2, "round_2": 2}
    df = load(
        str(base),
        extensions=["txt"],
        tag_dict=tag_dict,
        save_name="agg.csv",
        reset_files=True,
    )

    assert set(df["name"]) == {"baseline.txt", "round_1.txt", "round_2.txt"}
    assert set(df["tag"]) == {1, 2}
    assert set(df["text"]) == {"hello", "world", "!"}
    assert "layer_1" in df.columns
    assert os.path.exists(base / "agg.csv")


def test_load_existing_file_error(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "a.txt").write_text("a")
    load(str(base), save_name="out.csv", reset_files=True)
    with pytest.raises(FileExistsError):
        load(str(base), save_name="out.csv")


def test_load_csv_direct(tmp_path):
    df_in = pd.DataFrame({"a": [1, 2]})
    csv_path = tmp_path / "input.csv"
    df_in.to_csv(csv_path, index=False)
    df = load(str(csv_path), save_name="copy.csv", reset_files=True)
    assert (df == df_in).all().all()
    assert os.path.exists(tmp_path / "copy.csv")


def test_load_modality_paths(tmp_path):
    base = tmp_path / "media"
    base.mkdir()
    (base / "a.png").write_bytes(b"img")
    (base / "b.wav").write_bytes(b"aud")

    df_img = load(
        str(base),
        extensions=["png"],
        save_name="imgs.csv",
        reset_files=True,
        modality="image",
    )
    assert "path" in df_img.columns and "text" not in df_img.columns
    assert df_img["path"].tolist() == [str(base / "a.png")]

    df_aud = load(
        str(base),
        extensions=["wav"],
        save_name="aud.csv",
        reset_files=True,
        modality="audio",
    )
    assert "path" in df_aud.columns and "text" not in df_aud.columns
    assert df_aud["path"].tolist() == [str(base / "b.wav")]


def test_load_entity_behaves_like_text(tmp_path):
    path = tmp_path / "entity.txt"
    path.write_text("entity value")
    df = load(str(path), modality="entity", save_name="entity.csv", reset_files=True)
    assert df.loc[0, "text"] == "entity value"
    assert df.loc[0, "path"] == str(path)


def test_load_auto_detects_image(tmp_path):
    base = tmp_path / "imgs"
    base.mkdir()
    (base / "a.png").write_bytes(b"img")
    df = load(str(base), save_name="imgs.csv", reset_files=True)
    assert "text" not in df.columns
    assert df.loc[0, "path"].endswith("a.png")
