import asyncio
import os
import pandas as pd
from typing import Optional, Union, Any

from .tasks import (
    Rate,
    RateConfig,
    Classify,
    ClassifyConfig,
    Rank,
    RankConfig,
    Deidentifier,
    DeidentifyConfig,
    Codify,
    Extract,
    ExtractConfig,
    Paraphrase,
    ParaphraseConfig,
    Compare,
    CompareConfig,
)
from .utils.openai_utils import get_all_responses
from .utils.passage_viewer import view_coded_passages as _view_coded_passages
from .core.prompt_template import PromptTemplate
from .utils.parsing import safe_json
from .utils.openai_utils import get_all_embeddings
import unicodedata
import re
import math
from collections import defaultdict
import numpy as np

async def rate(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "ratings.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Rate`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = RateConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Rate(cfg).run(
        df,
        column_name,
        reset_files=reset_files,
    )

async def extract(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "extraction.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    types: Optional[dict[str, any]] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Extract`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = ExtractConfig(
        attributes=attributes,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Extract(cfg).run(
        df,
        column_name,
        reset_files=reset_files,
        types=types,
    )

async def classify(
    df: pd.DataFrame,
    column_name: str,
    *,
    labels: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    min_frequency: float = 0.6,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "classify_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Classify`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = ClassifyConfig(
        labels=labels,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        min_frequency=min_frequency,
        additional_instructions=additional_instructions or "",
        use_dummy=use_dummy,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Classify(cfg).run(
        df,
        column_name,
        reset_files=reset_files,
    )


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    use_dummy: bool = False,
    file_name: str = "deidentified.csv",
    max_words_per_call: int = 7500,
    guidelines: str = "",
    additional_guidelines: str = "",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Deidentifier`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = DeidentifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        max_words_per_call=max_words_per_call,
        guidelines=guidelines,
        additional_guidelines=additional_guidelines,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Deidentifier(cfg).run(df, column_name, grouping_column=grouping_column)


async def rank(
    df: pd.DataFrame,
    column_name: str,
    *,
    attributes: Union[dict[str, str], list[str]],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_rounds: int = 5,
    matches_per_round: int = 3,
    power_matching: bool = True,
    add_zscore: bool = True,
    compute_se: bool = True,
    learning_rate: float = 0.1,
    n_parallels: int = 750,
    use_dummy: bool = False,
    file_name: str = "rankings",
    reset_files: bool = False,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Rank`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = RankConfig(
        attributes=attributes,
        n_rounds=n_rounds,
        matches_per_round=matches_per_round,
        power_matching=power_matching,
        add_zscore=add_zscore,
        compute_se=compute_se,
        learning_rate=learning_rate,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        save_dir=save_dir,
        file_name=file_name,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Rank(cfg).run(
        df,
        column_name,
        reset_files=reset_files,
    )


async def codify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[dict[str, str]] = None,
    user_instructions: str = "",
    additional_instructions: str = "",
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    file_name: str = "coding_results.csv",
    reset_files: bool = False,
    debug_print: bool = False,
    use_dummy: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Codify`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    coder = Codify()
    return await coder.codify(
        df,
        column_name,
        categories=categories,
        user_instructions=user_instructions,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        additional_instructions=additional_instructions,
        n_parallels=n_parallels,
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        save_dir=save_dir,
        file_name=file_name,
        reset_files=reset_files,
        debug_print=debug_print,
        use_dummy=use_dummy,
    )


async def paraphrase(
    df: pd.DataFrame,
    column_name: str,
    *,
    instructions: str,
    save_dir: str,
    revised_column_name: Optional[str] = None,
    n_revisions: int = 1,
    file_name: str = "paraphrase_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    use_web_search: bool = False,
    n_parallels: int = 750,
    use_dummy: bool = False,
    reset_files: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    recursive_validation: bool = False,
    n_initial_candidates: int = 1,
    n_validation_candidates: int = 5,
    use_modified_source: bool = False,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Paraphrase`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = ParaphraseConfig(
        instructions=instructions,
        revised_column_name=revised_column_name,
        n_revisions=n_revisions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        use_web_search=use_web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        recursive_validation=recursive_validation,
        n_initial_candidates=n_initial_candidates,
        n_validation_candidates=n_validation_candidates,
        use_modified_source=use_modified_source,
        **cfg_kwargs,
    )
    return await Paraphrase(cfg).run(
        df,
        column_name,
        reset_files=reset_files,
    )


async def compare(
    df: pd.DataFrame,
    circle_column_name: str,
    square_column_name: str,
    *,
    save_dir: str,
    differentiation: bool = True,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "comparison_responses.csv",
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Compare`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = CompareConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        differentiation=differentiation,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Compare(cfg).run(
        df,
        circle_column_name,
        square_column_name,
        reset_files=reset_files,
    )


async def deduplicate(
    df: pd.DataFrame,
    on: str,
    *,
    save_dir: str,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "deduplication_responses.csv",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    use_embeddings: bool = True,
    group_size: int = 500,
    additional_instructions: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Deduplicate values in a single DataFrame column.

    Parameters
    ----------
    df:
        Input DataFrame.
    on:
        Name of the column to deduplicate.
    save_dir:
        Directory where intermediate CSV files will be saved.
    use_embeddings:
        When ``True`` (default), compute embeddings for the unique values and
        cluster them into groups no larger than ``group_size``.  When ``False``
        the unique values are sorted alphabetically and chunked.
    group_size:
        Maximum number of unique values to send in a single prompt.

    Returns
    -------
    DataFrame
        Original DataFrame with an additional ``mapped_{on}`` column
        containing the canonical values.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    df_proc = df.reset_index(drop=True).copy()

    # Step 1. Pre-deduplication ---------------------------------------------
    def _norm(s: Any) -> str:
        s = "" if s is None else str(s)
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = re.sub(r"[\W_]+", "", s).lower()
        return s

    norm_map: dict[str, str] = {}
    reps: list[str] = []
    collapsed: list[str] = []
    col_values = df_proc[on].astype(str).tolist()
    for val in col_values:
        key = _norm(val)
        rep = norm_map.setdefault(key, val)
        collapsed.append(rep)
        if rep == val:
            reps.append(val)
    df_proc["_pre_rep"] = collapsed

    unique_reps = reps

    # Step 2. Grouping strategy ---------------------------------------------
    groups: list[list[str]] = []
    if unique_reps:
        if use_embeddings:
            identifiers = [str(i) for i in range(len(unique_reps))]
            emb_path = os.path.join(save_dir, "dedup_embeddings.pkl")
            try:
                embeddings = await get_all_embeddings(
                    unique_reps,
                    identifiers=identifiers,
                    save_path=emb_path,
                    reset_file=reset_files,
                    n_parallels=n_parallels,
                    use_dummy=use_dummy,
                    verbose=False,
                )
                arr = np.array([embeddings[i] for i in identifiers])
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n_clusters = max(1, int(math.ceil(len(unique_reps) / group_size)))
                try:
                    from sklearn.cluster import KMeans

                    km = KMeans(n_clusters=n_clusters, n_init="auto")
                    labels = km.fit_predict(arr)
                    clusters: dict[int, list[str]] = defaultdict(list)
                    for lbl, item in zip(labels, unique_reps):
                        clusters[int(lbl)].append(item)
                    for items in clusters.values():
                        for i in range(0, len(items), group_size):
                            groups.append(items[i : i + group_size])
                except Exception:
                    for i in range(0, len(unique_reps), group_size):
                        groups.append(unique_reps[i : i + group_size])
            except Exception:
                for i in range(0, len(unique_reps), group_size):
                    groups.append(unique_reps[i : i + group_size])
        else:
            sorted_vals = sorted(unique_reps, key=lambda x: x.lower())
            for i in range(0, len(sorted_vals), group_size):
                groups.append(sorted_vals[i : i + group_size])

    # Step 3. Deduplication prompt ------------------------------------------
    template = PromptTemplate.from_package("deduplication_prompt.jinja2")
    prompts: list[str] = []
    ids: list[str] = []
    for idx, items in enumerate(groups):
        ids.append(f"group_{idx}")
        prompts.append(
            template.render(
                raw_terms="\n".join(items),
                additional_instructions=additional_instructions or "",
            )
        )

    csv_path = os.path.join(save_dir, file_name)
    if prompts:
        df_resp = await get_all_responses(
            prompts=prompts,
            identifiers=ids,
            n_parallels=n_parallels,
            model=model,
            save_path=csv_path,
            use_dummy=use_dummy,
            json_mode=True,
            reset_files=reset_files,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
            **kwargs,
        )
    else:
        df_resp = pd.DataFrame(columns=["Identifier", "Response"])

    resp_map = dict(zip(df_resp.get("Identifier", []), df_resp.get("Response", [])))
    mapping: dict[str, str] = {}
    for gid, items in zip(ids, groups):
        raw_resp = resp_map.get(gid, "")
        parsed = safe_json(raw_resp)
        if isinstance(parsed, dict):
            for rep, raws in parsed.items():
                if isinstance(raws, list):
                    for raw in raws:
                        mapping[str(raw)] = str(rep)
        for item in items:
            mapping.setdefault(item, item)

    # Step 4. Expansion back to original DataFrame -------------------------
    final_map = {rep: mapping.get(rep, rep) for rep in unique_reps}
    colname = f"mapped_{on}"
    df_out = df.copy()
    df_out[colname] = [final_map.get(rep, rep) for rep in df_proc["_pre_rep"]]

    return df_out


async def whatever(
    prompts: list[str],
    identifiers: list[str],
    *,
    save_dir: str,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    use_web_search: bool = False,
    n_parallels: int = 750,
    use_dummy: bool = False,
    reset_files: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper around :func:`get_all_responses` for arbitrary prompts.

    Results are saved to ``save_dir/file_name``.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    return await get_all_responses(
        prompts=prompts,
        identifiers=identifiers,
        save_path=save_path,
        model=model,
        json_mode=json_mode,
        use_web_search=use_web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reset_files=reset_files,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **kwargs,
    )


async def custom_prompt(
    prompts: list[str],
    identifiers: list[str],
    *,
    save_dir: str,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    use_web_search: bool = False,
    n_parallels: int = 750,
    use_dummy: bool = False,
    reset_files: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Backward compatible alias for :func:`whatever`."""

    return await whatever(
        prompts,
        identifiers,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        use_web_search=use_web_search,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reset_files=reset_files,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **kwargs,
    )


def view_coded_passages(
    df: pd.DataFrame,
    column_name: str,
    categories: Optional[Union[list[str], str]] = None,
    colab: bool = False,
):
    """Convenience wrapper for the passage viewer utility."""
    return _view_coded_passages(df, column_name, categories, colab=colab)
