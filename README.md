# GABRIEL

**GABRIEL** (Generalized Attribute Based Ratings Information Extraction Library) is an asynchronous toolkit for turning qualitative corpora into structured datasets with the help of modern GPT models.  It packages the prompting, batching, retry logic, and checkpointing you need to scale “ask the model” workflows to hundreds of thousands of passages, images, audio files, or entities.

📓 **Tutorial notebook**: https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing — start here for an end-to-end walkthrough.

## Table of contents

- [Why GABRIEL?](#why-gabriel)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Core capabilities](#core-capabilities)
- [Detailed usage](#detailed-usage)
- [Working with media and web search](#working-with-media-and-web-search)
- [Custom prompts and templates](#custom-prompts-and-templates)
- [Best practices for using GPT as a measurement tool](#best-practices-for-using-gpt-as-a-measurement-tool)
- [Saving, logging, and outputs](#saving-logging-and-outputs)
- [Development and testing](#development-and-testing)
- [Citation](#citation)

## Why GABRIEL?

Researchers are awash with qualitative data—parliamentary speeches, product reviews, interviews, historical archives—but need structured variables to test hypotheses.  GPT models are now capable of judging attributes (e.g. *patriotism*, *toxicity*, *policy focus*) and extracting facts with expert-level accuracy.  GABRIEL turns that raw capability into a robust measurement pipeline:

- 🧠 **Human-like comprehension at scale** – access GPT’s nuanced reasoning with simple configuration objects or one-line helper functions.
- 📊 **Quantitative outputs** – ratings, rankings, classifications, and extractions are returned as tidy DataFrames ready for statistical analysis.
- ⚙️ **Operational tooling** – automatic parallelism, retries, resumable runs, and template validation let you focus on research questions instead of API plumbing.
- 🧰 **Extensible workflows** – combine tasks for complex pipelines (e.g. deduplicate → codify → rank) or craft your own prompts via `gabriel.whatever`.

The guiding philosophy is parsimony: define attributes exactly as you would explain them to a human coder, then let GPT scale that measurement across your corpus.

## Installation

```bash
pip install gabriel

# or clone the prerelease repo and install locally
git clone https://github.com/allenai/GABRIEL-prerelease.git
cd GABRIEL-prerelease
pip install -e .
```

Set your API credentials before running real jobs:

```bash
export OPENAI_API_KEY="sk-..."
# Optional overrides if you use a compatible endpoint
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

All high-level helpers accept `use_dummy=True` for offline dry runs.

## Quick start

The snippet below mirrors the workflow in the tutorial notebook: rate Thanksgiving dishes on flavour attributes and save the results locally.

```python
import asyncio
import os
import pandas as pd
import gabriel

os.environ["OPENAI_API_KEY"] = "..."  # or rely on your shell environment

PATH = os.path.expanduser("~/Documents/gabriel_runs")
toy_data = pd.DataFrame({
    "entity": [
        "turkey",
        "pumpkin pie",
        "green bean casserole",
        "cornbread",
    ]
})

attributes = {
    "savory taste": "",
    "sweet taste": "",
    "tangy taste": "",
}

async def main():
    results = await gabriel.rate(
        toy_data,
        column_name="entity",
        attributes=attributes,
        save_dir=os.path.join(PATH, "toy_rate"),
        model="gpt-5-mini",
        n_runs=3,
        modality="entity",
        reset_files=True,
    )
    print(results.head())

asyncio.run(main())
```

Every task returns a `pandas.DataFrame` and writes raw model responses to `save_dir`.  Swap in other helpers (e.g. `gabriel.rank`, `gabriel.extract`) without changing your event loop setup.

## Core capabilities

| Function | Description | Typical use cases |
| --- | --- | --- |
| `gabriel.rate` | Score each row on 0–100 attributes with direct prompts. | Measuring concepts like sentiment, ideology, expertise, or tone. |
| `gabriel.rank` | Tournament-style comparisons that yield relative scores. | Distinguishing subtle ordering (e.g. most innovative technologies). |
| `gabriel.classify` | Multi-label or single-label tagging of passages. | Topic tagging, content moderation, checklist compliance. |
| `gabriel.extract` | Pull structured facts or metadata from text or images. | Inventor names, event dates, product attributes. |
| `gabriel.filter` | High-throughput boolean screen for huge candidate lists. | Keep only Wikipedia titles about technologies, shortlist case files. |
| `gabriel.codify` | Highlight passages that match qualitative codes. | Thematic analysis of interviews, policy documents, transcripts. |
| `gabriel.deidentify` | Replace PII with realistic, consistent placeholders. | Anonymising datasets before sharing or further analysis. |
| `gabriel.deduplicate` | Merge near-identical entities using embeddings + GPT. | Cleaning entity lists, removing redundant rows. |
| `gabriel.merge` | GPT-assisted fuzzy joins between dissimilar keys. | Matching records across messy schemas (job titles, product names). |
| `gabriel.compare` | Surface similarities or differences between paired texts. | Contrast persuasive vs. unpersuasive arguments, etc. |
| `gabriel.bucket` / `gabriel.discover` | Generate taxonomies and explain class differences. | Deriving emergent categories, exploratory feature discovery. |
| `gabriel.paraphrase` | Rewrite passages under explicit guidelines. | Creating alternative phrasings, redacting mentions, tone shifting. |
| `gabriel.debias` | Remove signal correlated with protected concepts. | Post-processing ratings to mitigate confounds. |
| `gabriel.whatever` | Minimal wrapper around `get_all_responses` for bespoke prompts. | Custom pipelines, experiments, prototyping. |

Each helper is built on a `Config` dataclass and a `.run()` coroutine in `gabriel.tasks`.  Use the top-level functions for convenience or drop down to the task classes for more control.

## Detailed usage

### Rating (`gabriel.rate` / `Rate`)
- Provide a DataFrame, the column to evaluate, and an attribute→definition mapping.
- Supports batching (`n_parallels`), multiple passes (`n_runs`), modality-specific prompts (`modality`), and optional reasoning traces (`reasoning_effort`, `reasoning_summary`).
- Saves intermediate CSVs (`file_name`, default `ratings.csv`).

### Ranking (`gabriel.rank` / `Rank`)
- Runs pairwise tournaments with Elo-style updates to capture fine-grained differences.
- Configure rounds, matches, learning rate, recursive refinement, and optional initial rating passes (`initial_rating_pass`).
- Ideal when absolute scales are ambiguous but relative ordering matters.

### Classification (`gabriel.classify` / `Classify`)
- Map label names to definitions; results include per-label probabilities and consensus columns.
- Optional differentiation mode asks the model to contrast “circle” vs. “square” passages for richer features.

### Extraction (`gabriel.extract` / `Extract`)
- Define attributes alongside descriptions of the desired outputs. Optional `types` enforce schemas (e.g. `{"year": "int"}`).
- Handy for building structured datasets out of biographies, product listings, or transcripts.

### Filtering (`gabriel.filter` / `Filter`)
- Specify a natural language condition (e.g. “return titles that describe an invention”).
- Processes short entities in large batches for efficiency; tune `entities_per_call`, `threshold`, and `shuffle` controls.

### Codifying (`gabriel.codify` / `Codify`)
- Provide qualitative codes and optional completion checks to ensure every passage was reviewed.
- Returns tidy tables plus helper functions like `gabriel.view` for rapid auditing, complete with clickable chips and numeric sliders to filter passages by categories, boolean labels, or rating bands.

### Deduplication and merging (`gabriel.deduplicate`, `gabriel.merge`)
- Combine embeddings with GPT adjudication to produce clean entity lists or fuzzy joins.
- Control chunk sizes, timeout behaviour, and auto-matching thresholds to fit your dataset.

### De-identification (`gabriel.deidentify` / `Deidentifier`)
- Supports multi-pass runs, consistent mapping columns, and strict “use only existing mappings” modes.
- Replace names, employers, addresses, or other PII with realistic substitutes guided by optional instructions.

### Paraphrasing (`gabriel.paraphrase` / `Paraphrase`)
- Rewrite text under constrained instructions, with optional recursive validation loops that select the best candidate paraphrase.

### Compare, bucket, and discover (`gabriel.compare`, `gabriel.bucket`, `gabriel.discover`)
- `compare` extracts similarities or contrasts between paired entries.
- `bucket` creates mutually exclusive categories from large term lists via iterative GPT voting.
- `discover` chains comparison, bucketing, and classification to surface explanatory features between two classes.

### Debiasing (`gabriel.debias` / `DebiasPipeline`)
- Estimate and remove confounding signals by codifying or rating auxiliary attributes, then regress them out of your measurement target.
- Configurable measurement/removal tasks, regression robustness, and stripping rules.

### Custom prompts (`gabriel.whatever` / `Whatever`)
- Accepts plain strings, lists, or DataFrames; automatically tracks identifiers and media attachments.
- Ideal for experiments that fall outside the packaged task templates while still benefiting from batching, retries, and persistent logs.

## Working with media and web search

`gabriel.utils.openai_utils.get_response` and `get_all_responses` underpin every task and accept image/audio attachments alongside text prompts.  Provide base64-encoded inputs directly or leverage helper utilities (`encode_image`, `encode_audio`) to load local files.  Models such as `gpt-4o-mini-audio-preview` can transcribe audio, while multimodal GPT-5 variants can reason over images.

For fact-finding prompts, pass `web_search=True` and optional `web_search_filters` (global) or `prompt_web_search_filters` (per prompt) to narrow results by geography, domain, or document type.  Filters accept an `allowed_domains` iterable and location hints (`city`, `country`, `region`, `timezone`, and the location `type`, typically `"approximate"`).  These controls are available through every top-level helper as keyword arguments.

## Custom prompts and templates

All task classes and convenience functions accept a `template_path` pointing to a Jinja2 template that mirrors the default variables for that task.  Templates are validated before use, so you get clear errors if a field is missing or renamed.  This makes it easy to customise instructions while keeping output parsing intact.

Prefer working directly with prompt strings?  Use `gabriel.whatever` and forward advanced flags such as `json_mode`, `web_search`, or `reasoning_effort`—they map straight through to `get_all_responses`.

## Best practices for using GPT as a measurement tool

- **Treat GPT outputs as measurements**, then apply the same statistical discipline you would with survey or lab data (hold-out sets, robustness checks, documentation).
- **Explore your corpus first**: read samples, pilot attributes in ChatGPT, and ensure definitions are precise yet concise.
- **Guard against p-hacking** by logging every attribute you try and validating on reserved data splits when running large grids of measurements.
- **Parallelise liberally**: GABRIEL handles concurrency and retry logic, so you can scale to large corpora quickly.
- **Checkpoint and audit**: intermediate CSVs, JSON logs, and helper viewers (`gabriel.view`) make it easy to inspect outputs and rerun failed batches.

These principles—and many richer examples—are documented in the tutorial notebook linked above.

## Saving, logging, and outputs

Every run expands `save_dir` (supporting `~` and environment variables), creates the directory if needed, and writes:

- `file_name` CSV/Parquet outputs with structured results.
- A `responses` subfolder containing raw model payloads from `get_all_responses`.
- Configuration metadata so you can reproduce experiments.

You can resume partially completed runs by leaving `reset_files=False` (the default) or start fresh by passing `reset_files=True`.

## Development and testing

Install development extras and run the test suite:

```bash
pip install -e .[dev]
pytest
```

Tests rely on dummy responses, so no API key is required.  Linting and type checks are available via `ruff` and `mypy` when the extras are installed.

## Citation

If you use GABRIEL in your research, please cite:

> The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2023). <https://github.com/elliottmokski/GABRIEL-distribution>
