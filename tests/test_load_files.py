import os
import pandas as pd
import pytest

from gabriel import load_files


def test_load_files_basic(tmp_path):
    base = tmp_path / "data"
    sub = base / "round"
    base.mkdir()
    sub.mkdir()

    (base / "baseline.txt").write_text("hello")
    (sub / "round_1.txt").write_text("world")
    (sub / "round_2.txt").write_text("!")

    tag_dict = {"baseline": 1, "round_1": 2, "round_2": 2}
    df = load_files(
        str(base),
        extensions=["txt"],
        tag_dict=tag_dict,
        save_name="agg.csv",
        reset_files=True,
    )

    assert set(df["name"]) == {"baseline.txt", "round_1.txt", "round_2.txt"}
    assert "layer_1" in df.columns
    assert set(df["tag"]) == {1, 2}
    assert os.path.exists(base / "agg.csv")


def test_load_files_existing_file_error(tmp_path):
    base = tmp_path / "data"
    base.mkdir()
    (base / "a.txt").write_text("a")
    load_files(str(base), save_name="out.csv", reset_files=True)
    with pytest.raises(FileExistsError):
        load_files(str(base), save_name="out.csv")


def test_load_files_csv_direct(tmp_path):
    df_in = pd.DataFrame({"a": [1, 2]})
    csv_path = tmp_path / "input.csv"
    df_in.to_csv(csv_path, index=False)
    df = load_files(str(csv_path), save_name="copy.csv", reset_files=True)
    assert (df == df_in).all().all()
    assert os.path.exists(tmp_path / "copy.csv")


def test_load_files_modality_paths(tmp_path):
    base = tmp_path / "media"
    base.mkdir()
    (base / "a.png").write_bytes(b"img")
    (base / "b.wav").write_bytes(b"aud")

    df_img = load_files(
        str(base),
        extensions=["png"],
        save_name="imgs.csv",
        reset_files=True,
        modality="image",
    )
    assert "image_path" in df_img.columns and "content" not in df_img.columns
    assert df_img["name"].tolist() == ["a.png"]

    df_aud = load_files(
        str(base),
        extensions=["wav"],
        save_name="aud.csv",
        reset_files=True,
        modality="audio",
    )
    assert "audio_path" in df_aud.columns and "content" not in df_aud.columns
    assert df_aud["name"].tolist() == ["b.wav"]
