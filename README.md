# GABRIEL

**GABRIEL** (Generalized Attribute Based Ratings Information Extraction Library) turns messy qualitative corpora into analysis-ready datasets with GPT. It handles prompting, batching, retries, checkpointing, and audit trails so you can treat “ask the model” workflows like any other measurement instrument. From rating rhetoric across a million speeches to matching product catalogs, you focus on the research question while GABRIEL handles the operations.

📓 **Tutorial notebook** (start here!): https://colab.research.google.com/drive/1RMUeAWACpViqiUMlPMMwPTKyGU-OX756?usp=sharing — also available as `tutorial_notebook.ipynb` in this repo if you’d like to download and run it locally.

## Table of contents

- [Why GABRIEL?](#why-gabriel)
- [What can you do with GABRIEL?](#what-can-you-do-with-gabriel)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Task highlights](#task-highlights)
- [Detailed usage](#detailed-usage)
- [Multimodal data and web search](#multimodal-data-and-web-search)
- [Custom prompts and model routing](#custom-prompts-and-model-routing)
- [Saving, logging, and resuming](#saving-logging-and-resuming)
- [Development and testing](#development-and-testing)
- [Citation](#citation)

## Why GABRIEL?

Most of the evidence social scientists and analysts care about lives in unstructured formats: interviews, speeches, transcripts, product photos, archival scans. Modern GPT models can judge attributes, extract facts, and reason about this material with high fidelity, but building robust pipelines is still tedious. GABRIEL provides:

- 🧠 **Human-level comprehension on demand** – express the attribute the way you would brief a human coder; GABRIEL packages the prompt, context, and retries for you.
- 📊 **Quantitative outputs** – ratings (0–100), grounded comparisons, classifications, and structured extractions return as tidy DataFrames with reproducible settings.
- ⚙️ **Operational tooling** – automatic parallelism (hundreds of concurrent calls), resumable runs, raw response logs, and helper UIs make it safe to scale to very large corpora.
- 🧰 **Extensibility** – swap instructions with `additional_instructions`, bring your own templates, or drop down to `gabriel.whatever` + custom `response_fn` for bespoke prompts while still reusing the infrastructure.

The tutorial notebook walks through these ideas step-by-step—from setting up an API key to running multimodal analyses—so skim this README, then dive into the notebook for the full guided tour.

## What can you do with GABRIEL?

### A) Measure attributes on qualitative data

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.rate` | Asks GPT to score each text / image / audio / item on natural language attributes. Output = 0--100 rating. | Measure “patriotic rhetoric” in a speech; “toxicity” of tweets; “luxury” in ad images. |
| `gabriel.rank` | Pairwise comparisons between texts yields ELO-like attribute ratings. Output = grounded, relative z scores for each text. | Rank technologies by “bulkiness” or artworks by “fine brushwork”. |
| `gabriel.classify` | Classifies texts / images / audio / items on whether provided labels apply. Output = one or more classes per item. | Tag news articles, product photos, or interview clips into topical categories. |
| `gabriel.extract` | Structured fact extraction on each item. Output = string / numeric values. | For each product, provide the “company”, “CEO”, and “year of invention”. |
| `gabriel.discover` | Discovers natural language features which discriminate two classes of data. | Identify what distinguishes 5 star vs. 1 star reviews or successful vs. failed startups. |

### B) Clean data

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.merge` | Creates crosswalks. Output = merged table with GPT-matched identifiers. | Match two distinct job title directories; link patent titles to product names. |
| `gabriel.deduplicate` | Detects conceptual duplicates. Maps all duplicates to one representative term. | Collapse “F-18”, “Super Hornet Fighter Jet”, “f-18 hornet” into “F-18”. |
| `gabriel.filter` | High-throughput boolean screening. Outputs items which meet natural language condition. | Subset 18M Wikipedia titles to only technologies. |
| `gabriel.deidentify` | Replaces PII with realistic, consistent fake PII. Outputs anonymized text + mapping. | Replace names, employers, addresses before sharing interview corpora. |

### C) Helper tools

| Function | Purpose & Output Scale | Example Use |
| --- | --- | --- |
| `gabriel.codify` | Passage coding: highlights snippets in text that match qualitative codes. | Flag sentences about “economic insecurity” in speeches; “stressors” mentioned in interview. |
| `gabriel.compare` | Identifies similarities / differences between paired items. Output = list of differences. | Contrast op-eds from different districts; compare two ad campaigns. |
| `gabriel.bucket` | Builds taxonomies from many terms. Output = bucket/cluster labels. | Group technologies, artworks, or HR complaints into emergent categories. |
| `gabriel.seed` | Enforces a representative distribution / diversity of seeds. | Initialize unique personas that match US population distribution. |
| `gabriel.ideate` | Generates many novel scientific theories and filters the cream of the crop. | Procure novel theories on inflation for potential research. |
| `gabriel.debias` | Post-process measurements to remove inference bias. | Ensure GPT isn't guessing climate opinions in speeches based on general political lean. |
| `gabriel.load` | Prepares a folder of text / image / audio files into a spreadsheet for use in GABRIEL. | Image directory converted into spreadsheet of file paths. |
| `gabriel.view` | UI to view sample texts with ratings / passage coding. | Spot-check classify / rating outputs; view coded passages. |
| `gabriel.paraphrase` | Rewrites texts consistently per instructions. | Summarize earnings call transcripts to remove company specifics. |
| `gabriel.whatever` | Run any GPT prompts, but leverage GABRIEL's parallelization / checkpointing. | Any set of prompts; slots into any pipeline. |

## Installation

```bash
pip install gabriel

# or work from this prerelease repo
git clone https://github.com/hemanth-asirvatham/GABRIEL-prerelease.git
cd GABRIEL-prerelease
pip install -e .

# install extra tooling for local development and tests
pip install -e .[dev]
pip install -r requirements-dev.txt
```
You can optionally sanity-check the install with a quick version print:

```bash
gabriel --version  # optional check
```

Before running real jobs, point the helpers to your GPT endpoint:

```bash
export OPENAI_API_KEY="sk-..."
# Optional if you proxy the OpenAI API
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Every task also accepts `use_dummy=True` for offline dry runs (the tutorial uses this to demonstrate workflows without making API calls).

## Quick start

The tutorial notebook walks through many complete projects; here’s the minimal rating flow the notebook starts with. Paste this into Colab or a notebook cell so you can use `await` directly:

```python
import os
import pandas as pd

import gabriel

PATH = os.path.expanduser("~/Documents/gabriel_runs")
toy_data = pd.DataFrame(
    {
        "entity": [
            "turkey",
            "pumpkin pie",
            "green bean casserole",
            "cornbread",
        ]
    }
)

attributes = {
    "savory taste": "How savory the dish is",
    "sweet taste": "Dessert-like sweetness",
    "tangy taste": "Notes of tartness or acidity",
}

rate_results = await gabriel.rate(
    toy_data,
    column_name="entity",
    attributes=attributes,
    save_dir=os.path.join(PATH, "toy_rate"),
    model="gpt-5-mini",
    n_runs=2,
    modality="entity",
    reset_files=True,
)
rate_results.head()
```

The helper returns a `pandas.DataFrame` with one column per attribute and writes raw model responses + configs to `save_dir`. Running the same code in a plain Python script just requires wrapping the coroutine with `asyncio.run(...)`.

## Task highlights

All helpers share the same ergonomics: pass a DataFrame (or folder path prepared with `gabriel.load`), choose the relevant text/image/audio column, and specify task-specific settings. Useful knobs such as `model`, `n_parallels` (concurrency), `n_runs`, `reasoning_effort`, `reasoning_summary`, `additional_instructions`, and `template_path` are available everywhere.

### 1. Measurement primitives
- **`gabriel.rate`** – assign 0–100 scores per attribute. Supports batching attributes, reasoning traces, and modalities (`text`, `entity`, `image`, `audio`, `web`).
- **`gabriel.rank`** – tournament-style pairwise comparisons that yield grounded relative z-scores. Configure `n_rounds`, `matches_per_round`, and `recursive=True` to iteratively surface the best performers (as in the ideation example).
- **`gabriel.classify`** – multi- or single-label tagging with label definitions, consensus columns, and optional differentiation prompts when you want richer rationales.
- **`gabriel.extract`** – turn unstructured passages into tidy tables by naming attributes and optional data types; great for biographies, filings, and multimodal product cards.
- **`gabriel.discover`** – contrast two labeled corpora to surface discriminating features, combining compare → bucket → classify under the hood.

### 2. Qualitative coding, review, and taxonomy building
- **`gabriel.filter`** screens very large candidate lists with boolean conditions (e.g., keep only inventions) and tunable thresholds.
- **`gabriel.codify`** highlights snippets that match qualitative codes and pairs with **`gabriel.view`** to audit coded passages with chips and sliders.
- **`gabriel.view`** reads any saved run (rate/classify/codify/etc.) and renders interactive tables so you can spot-check results in the browser.
- **`gabriel.compare`** explains differences or similarities for paired items (op-eds, policies, drafts).
- **`gabriel.bucket`** groups terms or entities into emergent taxonomies, which can then be fed back into rate/classify calls.

### 3. Data prep, linkage, and rewriting
- **`gabriel.merge`** and **`gabriel.deduplicate`** combine embeddings with GPT adjudication to produce clean entity lists or fuzzy joins.
- **`gabriel.deidentify`** replaces PII with realistic stand-ins, respecting grouping columns and optionally reusing an existing mapping.
- **`gabriel.paraphrase`** rewrites passages under strict guidance (e.g., remove brand mentions) with optional recursive validation loops.
- **`gabriel.debias`** regresses out unwanted signals by measuring confounds (via rate/codify) and stripping them from your primary measurement.

### 4. Discovery, seeding, and ideation pipelines
- **`gabriel.seed`** generates diverse seed entities/personas, enforcing representation constraints before you collect more data.
- **`gabriel.ideate`** brainstorms hundreds of candidate hypotheses, then filters/ranks them (often chained with `gabriel.rank`).

### 5. Data ingestion and automation
- **`gabriel.load`** walks a folder of text/image/audio files and produces a spreadsheet with identifiers + file paths so downstream tasks can attach those modalities.
- **`gabriel.whatever`** accepts fully custom prompts, attachments, and advanced flags (`json_mode`, `web_search`, `response_fn`) while reusing retries, parallelism, and checkpointing.

## Detailed usage

The tutorial notebook shows full projects end-to-end; the summaries below serve as a quick reference to the knobs each helper exposes.

### Measurement primitives
- **Rating (`gabriel.rate` / `Rate`)** – Provide a DataFrame, the column to evaluate, and an attribute→definition mapping. Supports batching (`n_parallels`), multiple passes (`n_runs`), modality-specific prompts (`modality`), and optional reasoning traces (`reasoning_effort`, `reasoning_summary`). Saves intermediate CSVs (`file_name`, default `ratings.csv`) under `save_dir` alongside raw responses.
- **Ranking (`gabriel.rank` / `Rank`)** – Runs pairwise tournaments with Elo-style updates to capture fine-grained differences. Configure `n_rounds`, `matches_per_round`, `learning_rate`, and `recursive=True` to iteratively refine the leaderboard. Use `initial_rating_pass` or `initial_rating_field` to seed scores from a prior `rate` run when helpful.
- **Classification (`gabriel.classify` / `Classify`)** – Map label names to definitions; results include per-label probabilities and consensus columns. Optional differentiation mode asks the model to contrast close labels for richer rationales. Works well for multimodal tagging when paired with `modality="image"/"audio"` and for mutually exclusive labels using `additional_instructions`.
- **Extraction (`gabriel.extract` / `Extract`)** – Define attributes alongside descriptions of the desired outputs; optional `types` enforce schemas (e.g. `{ "year": "int" }`). Handles JSON mode and nested schemas for complex cards (e.g., product + specs + price). Combine with `gabriel.load` to attach file paths for multimodal product cards.
- **Discovery (`gabriel.discover` / `Discover`)** – Chains compare → bucket → classify to surface discriminating features across two classes of data. Configure reference columns, feature counts, and optional follow-up classification thresholds.

### Qualitative coding, review, and taxonomy building
- **Filtering (`gabriel.filter` / `Filter`)** – Screens huge candidate lists with a natural-language condition; tune `entities_per_call`, `threshold`, and `shuffle` to balance recall vs. throughput.
- **Passage coding (`gabriel.codify` / `Codify`)** – Highlights snippets that match qualitative codes and writes token-level spans you can audit later. Works hand-in-hand with `gabriel.view` for UI-based inspection.
- **Interactive review (`gabriel.view` / `View`)** – Loads any saved output directory and exposes sliders, chips, and search so analysts can examine results without leaving the browser.
- **Comparisons (`gabriel.compare` / `Compare`)** – Provides bullet-point similarities/differences for paired entries (e.g., draft A vs. draft B). Adjust `perspective` or `comparison_prompt` to focus on specific axes.
- **Taxonomies (`gabriel.bucket` / `Bucket`)** – Groups long lists of entities or terms into emergent categories; configure `n_buckets`, `n_examples_per_bucket`, and optional seeding labels. Outputs can be piped back into `classify` or `rate`.

### Data prep, linkage, rewriting, and bias correction
- **Entity cleanup (`gabriel.merge` / `Merge` and `gabriel.deduplicate` / `Deduplicate`)** – Combine embeddings, string similarity, and GPT adjudication to produce clean entity lists or fuzzy joins. Control chunk sizes, timeout behavior, and auto-matching thresholds; outputs include mapping columns you can feed to downstream analyses.
- **Privacy-preserving transforms (`gabriel.deidentify` / `Deidentify`)** – Replace PII with realistic stand-ins, respecting grouping columns and optionally reusing existing mappings to maintain consistency over time.
- **Guided rewriting (`gabriel.paraphrase` / `Paraphrase`)** – Rewrite passages under strict guidance (e.g., remove brand mentions or standardize tone) with optional recursive validation loops.
- **Bias correction (`gabriel.debias` / `Debias`)** – Measure confounds (typically with `rate`/`codify`), regress them out of your main measurements, and export adjusted scores alongside diagnostics.

### Discovery, seeding, and ideation pipelines
- **Idea generation (`gabriel.ideate` / `Ideate`)** – Provide a topic, number of ideas, and optional filtering heuristics; the helper generates candidates in parallel, logs intermediate vetting, and can be chained to `rank` for tournament selection.
- **Diversity seeding (`gabriel.seed` / `Seed`)** – Specify the traits you need in your seed set (e.g., demographic quotas, experience levels). The helper enforces representativeness while surfacing rich descriptions to feed downstream measurement.

### Data ingestion and custom prompts
- **Loading media (`gabriel.load` / `Load`)** – Walk a folder of text/image/audio files and produce a spreadsheet with identifiers + file paths so every downstream task can attach those modalities. Handy for multimodal runs and for the county-level web search example when you later rate the generated reports.
- **Fully custom prompts (`gabriel.whatever` / `Whatever`)** – Accepts arbitrary prompt text, attachments, and settings such as `json_mode`, `response_fn`, or `web_search`. You keep retries, rate-limit handling, parallelism, and persistent logging even when you bypass the built-in templates.

## Multimodal data and web search

All measurement helpers accept a `modality` argument (`text`, `entity`, `image`, `audio`, or `web`). When working with folders of media, run `gabriel.load` first to expand files into rows with clean IDs. Set `web_search=True` (plus optional domain/location filters) on `gabriel.whatever` or pass `modality="web"` to `gabriel.rate`/`gabriel.extract` when you want GPT to gather the relevant context before answering. The same batching, retries, and checkpointing apply regardless of modality, and the tutorial’s county-level example shows how to chain web search → rating → mapping in one flow.

## Custom prompts and model routing

Tweak any task by:

1. **Appending extra instructions** – Pass natural-language clarifications through `additional_instructions` (as in the classification tutorial example) to mandate mutually exclusive labels, share rubrics, or supply few-shot demonstrations.
2. **Supplying your own template** – Provide a Jinja file via `template_path` that uses the same variables as our default prompts; you can keep all retries/checkpointing while tailoring the instructions or formatting.
3. **Going fully custom** – Use `gabriel.whatever` when you need bespoke prompts, attachments, or `response_fn` logic. You still benefit from automated retries, rate-limit handling, persistent logging, and options like `web_search=True`.

## Saving, logging, and resuming

Each run expands `save_dir` (tilde and environment variables supported), writes your structured output (`file_name` CSV/Parquet), and saves raw model payloads under `responses/` together with metadata so you can audit later. Leave `reset_files=False` (default) to resume partially completed runs; delete the folder or pass `reset_files=True` to start fresh. `gabriel.view` reads these outputs to provide a lightweight UI for spot checks, and helpers like `gabriel.utils.mapmaker.MapMaker` can consume the same files for downstream visualization.

## Development and testing

Install development extras and run tests:

```bash
pip install -e .[dev]
pytest
```

Tests rely on the built-in dummy responses, so no API key is necessary. Linting and type checks (`ruff`, `mypy`) are also included in the dev extras.

## Citation

If you use GABRIEL in your research, please cite:

> The Generalized Attribute Based Ratings Information Extraction Library (GABRIEL). Hemanth Asirvatham and Elliott Mokski (2025). <https://github.com/hemanth-asirvatham/GABRIEL-prerelease>
