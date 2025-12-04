"""Tests for custom prompt template loading and validation."""

from importlib import resources
from pathlib import Path

import pytest

from gabriel.core.prompt_template import PromptTemplate


def test_custom_template_matches_variables(tmp_path):
    """A template with identical variables should load without error."""

    ref = resources.files("gabriel.prompts").joinpath("ratings_prompt.jinja2").read_text(
        encoding="utf-8"
    )
    custom_path = tmp_path / "custom.jinja2"
    custom_path.write_text(ref, encoding="utf-8")

    tmpl = PromptTemplate.from_file(str(custom_path), reference_filename="ratings_prompt.jinja2")
    assert tmpl.text == ref


def test_custom_template_variable_mismatch(tmp_path):
    """Missing required variables should raise a descriptive error."""

    wrong_path = tmp_path / "wrong.jinja2"
    wrong_path.write_text("{{ text }}", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        PromptTemplate.from_file(str(wrong_path), reference_filename="ratings_prompt.jinja2")
    msg = str(excinfo.value)
    assert "missing variables" in msg or "unexpected variables" in msg


def test_custom_template_allows_minimum_required(tmp_path):
    """Templates with only required fields should be accepted with a warning."""

    custom_path = tmp_path / "loose.jinja2"
    custom_path.write_text("{{ text }} || {{ attributes }}", encoding="utf-8")

    with pytest.warns(UserWarning, match="Custom template variable mismatch"):
        tmpl = PromptTemplate.from_file(str(custom_path), reference_filename="ratings_prompt.jinja2")
    rendered = tmpl.render(text="hello", attributes={"a": "1"})
    assert "hello" in rendered
    assert "a" in rendered
