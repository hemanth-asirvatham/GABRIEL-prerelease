# GABRIEL

TRIAL NOTEBOOK: https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing. See this notebook for the most updated example set.

GABRIEL (Generalized Attribute Based Ratings Information Extraction Library) is a collection of utilities for running large language model driven analyses.  The library provides high level tasks such as passage rating, text classification, de‑identification, regional report generation and several Elo style ranking utilities.

The current `src` directory contains a cleaned up and asynchronous implementation.  Each task exposes an easy to use `run()` coroutine and sensible configuration dataclasses. 

## Quick Start

```python
from gabriel.tasks import Rate, RateConfig

cfg = RateConfig(
    attributes={"clarity": "How understandable is the text?"},
    save_path="ratings.csv",
    use_dummy=True  # set to False to call the OpenAI API
)

texts = ["This is an example passage"]
ratings = asyncio.run(Rate(cfg).run(texts))
print(ratings)
```

Each task returns a `pandas.DataFrame` and saves raw responses to disk.  Set `use_dummy=False` and provide your OpenAI credentials via the `OPENAI_API_KEY` environment variable to perform real API calls.
If your OpenAI-compatible service uses a different endpoint, set `OPENAI_BASE_URL`
or pass a `base_url` argument to override the default API URL.

### Image and audio inputs

`get_response` and `get_all_responses` can include images or audio with your prompts. Pass `images` and/or `audio` to `get_response` or supply `prompt_images` and `prompt_audio` mappings to `get_all_responses`. Images should be base64 strings and audio entries should be dictionaries containing `data` and `format`. The notebook‑ready cells below show how to fetch media from the web and make a request using `await`:

#### Image example

```python
import aiohttp, base64
from gabriel.utils import get_response

# Download an image from the internet
async with aiohttp.ClientSession() as session:
    async with session.get(
        "https://raw.githubusercontent.com/github/explore/main/topics/python/python.png"
    ) as resp:
        img_bytes = await resp.read()
img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Ask the model about the picture
responses, _ = await get_response(
    "What logo is this?", images=[img_b64], use_dummy=True
)
print(responses[0])
```

#### Audio example

```python
import aiohttp, base64
from gabriel.utils import get_response

# Download an audio clip
async with aiohttp.ClientSession() as session:
    async with session.get(
        "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav"
    ) as resp:
        audio_bytes = await resp.read()
audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

# Transcribe the clip
responses, _ = await get_response(
    "Transcribe the clip",
    audio=[{"data": audio_b64, "format": "wav"}],
    model="gpt-4o-mini-audio-preview",
    use_dummy=True,
)
print(responses[0])
```

Images are provided as base64 strings, while audio items are dictionaries with `data` and `format`. Helper functions `encode_image` and `encode_audio` are available for local files.

### Per-prompt web search filters

`get_all_responses` accepts a `prompt_web_search_filters` mapping for prompt-specific web search hints. The keys should match the prompt identifiers passed to `get_all_responses` and each value should mirror the structure of `web_search_filters` (for example including `city`, `region`, `country`, `timezone`, `type`, or `allowed_domains`). These per-prompt settings are merged with any global `web_search_filters` before each request, so you can keep a shared domain whitelist while pulling location hints from a DataFrame column on a row-by-row basis. See `gabriel.utils.openai_utils` for the full normalisation logic.

### Custom prompts with `gabriel.whatever`

Use `gabriel.whatever` when you need a thin wrapper around `get_all_responses`. The helper can work with a single string prompt, a list of prompts, or a DataFrame column. When a DataFrame is supplied you can optionally specify:

- `column_name` – which column contains the prompt text (required for DataFrame input).
- `identifier_column` – supply unique IDs; otherwise short SHA1 identifiers are generated.
- `image_column` / `audio_column` – columns containing media to be encoded automatically via `load_image_inputs` and `load_audio_inputs`.
- `web_search_filters` – pass a dictionary whose values may be column names; for example `{"city": "city_col", "allowed_domains": "domains"}` will read the appropriate cell for every prompt and build a `prompt_web_search_filters` map automatically.

Results are saved to `save_dir/file_name` and all other keyword arguments are forwarded to `get_all_responses`, making it easy to reuse advanced features like JSON mode, batches, or custom tool definitions.
### Custom prompt templates

All task classes and the high-level API functions accept a `template_path`
argument. Supply the path to a Jinja2 file that declares the same variables
as the built-in template for that task and it will be used instead. Variable
sets are validated before use and a helpful error is raised if a template is
missing or introduces unexpected parameters.

## Tasks

### `Rate`
Rate passages on a set of numeric attributes.  The task builds prompts using `gabriel.prompts.ratings_prompt.jinja2` and parses the JSON style output into a `dict` for each passage.

Key options (see `RateConfig`):
- `attributes` – mapping of attribute name to description.
- `model` – model name (default `gpt-5-mini`).
- `n_parallels` – number of concurrent API calls.
- `save_path` – CSV file for intermediate results.
- `rating_scale` – optional custom rating scale text. If omitted, the default 0–100 scale from the template is used.

### `Classify`
Classify passages into boolean labels.  Uses a prompt in `basic_classifier_prompt.jinja2` and expects JSON `{label: true/false}` responses.

Options include the label dictionary, output directory, model and an optional maximum timeout.  Results are joined back onto the input DataFrame with one column per label.

### `Deidentifier`
Iteratively remove identifying information from text.  Texts are split into manageable chunks and the model returns JSON replacement mappings which are applied across all rows.

Configuration allows controlling the maximum words per call, LLM model and any additional guidelines for the prompt.

### `EloRater`
Pairwise Elo / Bradley–Terry rating of items across any set of attributes.  Prompts are built from the `rankings_prompt.jinja2` template and include explicit support for win/loss/draw outcomes.

`EloConfig` controls the number of rounds, matches per round, rating method (Elo, BT or Plackett–Luce), parallelism and more.  The final DataFrame includes rating, optional standard error and z-score columns.

### `RecursiveEloRater`
Higher level orchestrator that repeatedly applies `EloRater` on progressively filtered subsets.  Items can optionally be rewritten between stages and cumulative scores are tracked across recursion steps.

### `Regional`
Generate short reports for topics across regions (for example counties or states).  Results are stored in a wide DataFrame with one column per topic.

### `CountyCounter`
Convenience wrapper that chains a `Regional` run followed by Elo rating of each regional report.  Optionally produces Plotly choropleth maps if FIPS codes are provided.

## Utilities

The `gabriel.utils` module contains helpers for interacting with the OpenAI API, rendering prompt templates and creating visualisations.  The `OpenAIClient` class in `gabriel.core` provides a minimal asynchronous interface for customised pipelines.

## Running the Tests

Install the development dependencies and run `pytest`:

```bash
pip install -e .[dev]
pytest
```

All tests use `use_dummy=True` so no API key is required.

## Citation

If you use GABRIEL in your research, please cite:

> The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2023). <https://github.com/elliottmokski/GABRIEL-distribution>
