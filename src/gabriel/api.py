import asyncio
import os
import pandas as pd
from typing import Dict, Optional, Union

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
    CodifyConfig,
    Extract,
    ExtractConfig,
    Paraphrase,
    ParaphraseConfig,
    Compare,
    CompareConfig,
    Merge,
    MergeConfig,
    Deduplicate,
    DeduplicateConfig,
    Bucket,
    BucketConfig,
    Discover,
    DiscoverConfig,
    Filter,
    FilterConfig,
)
from .utils.openai_utils import get_all_responses
from .utils.passage_viewer import view_coded_passages as _view_coded_passages
from .core.prompt_template import PromptTemplate

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
    template_path: Optional[str] = None,
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
    return await Rate(cfg, template_path=template_path).run(
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
    template_path: Optional[str] = None,
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
    return await Extract(cfg, template_path=template_path).run(
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
    template_path: Optional[str] = None,
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
    return await Classify(cfg, template_path=template_path).run(
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
    template_path: Optional[str] = None,
    reset_files: bool = False,
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
    return await Deidentifier(cfg).run(
        df,
        column_name,
        grouping_column=grouping_column,
        reset_files=reset_files,
    )

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
    template_path: Optional[str] = None,
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
    return await Rank(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
    )


async def codify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    categories: Optional[Dict[str, str]] = None,
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
    template_path: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Codify`."""
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = CodifyConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        debug_print=debug_print,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Codify(cfg, template_path=template_path).run(
        df,
        column_name,
        categories=categories,
        additional_instructions=additional_instructions,
        reset_files=reset_files,
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
    template_path: Optional[str] = None,
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
    return await Paraphrase(cfg, template_path=template_path).run(
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
    differentiate: bool = True,
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
    template_path: Optional[str] = None,
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
        differentiate=differentiate,
        additional_instructions=additional_instructions or "",
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Compare(cfg, template_path=template_path).run(
        df,
        circle_column_name,
        square_column_name,
        reset_files=reset_files,
    )


async def bucket(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "bucket_definitions.csv",
    bucket_count: int = 10,
    differentiate: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    template_path: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Bucket`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = BucketConfig(
        bucket_count=bucket_count,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Bucket(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
    )


async def discover(
    df: pd.DataFrame,
    *,
    column_name: Optional[str] = None,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 1,
    min_frequency: float = 0.6,
    bucket_count: int = 10,
    differentiate: bool = True,
    max_words_per_call: int = 1000,
    max_categories_per_call: int = 8,
    n_terms_per_prompt: int = 250,
    repeat_bucketing: int = 5,
    repeat_voting: int = 25,
    next_round_frac: float = 0.25,
    top_k_per_round: int = 1,
    raw_term_definitions: bool = True,
    use_dummy: bool = False,
    modality: str = "text",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    reset_files: bool = False,
    **cfg_kwargs,
) -> Dict[str, pd.DataFrame]:
    """Convenience wrapper for :class:`gabriel.tasks.Discover`.

    Returns intermediate DataFrames from each step of the discovery pipeline.
    When ``circle_column_name`` and ``square_column_name`` are provided,
    classification is performed twice (once evaluating the circle entry and
    once the square entry) using prompts that contain both entries.
    A ``summary`` key is included in the result describing label prevalence
    differences between the two directions (``difference_pct`` expresses
    circle minus square in percentage points).
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = DiscoverConfig(
        save_dir=save_dir,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        min_frequency=min_frequency,
        bucket_count=bucket_count,
        additional_instructions=additional_instructions,
        differentiate=differentiate,
        max_words_per_call=max_words_per_call,
        max_categories_per_call=max_categories_per_call,
        n_terms_per_prompt=n_terms_per_prompt,
        repeat_bucketing=repeat_bucketing,
        repeat_voting=repeat_voting,
        next_round_frac=next_round_frac,
        top_k_per_round=top_k_per_round,
        raw_term_definitions=raw_term_definitions,
        use_dummy=use_dummy,
        modality=modality,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        **cfg_kwargs,
    )
    return await Discover(cfg).run(
        df,
        column_name=column_name,
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
    )


async def deduplicate(
    df: pd.DataFrame,
    on: str,
    *,
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    n_runs: int = 3,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "deduplicate_responses.csv",
    use_embeddings: bool = True,
    group_size: int = 500,
    max_timeout: Optional[float] = None,
    template_path: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Deduplicate`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = DeduplicateConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        max_timeout=max_timeout,
        additional_instructions=additional_instructions,
        use_embeddings=use_embeddings,
        group_size=group_size,
        **cfg_kwargs,
    )
    return await Deduplicate(cfg, template_path=template_path).run(
        df,
        on=on,
        reset_files=reset_files,
    )


async def merge(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    save_dir: str,
    on: Optional[str] = None,
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    how: str = "left",
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-nano",
    n_parallels: int = 750,
    n_runs: int = 1,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "merge_responses.csv",
    use_embeddings: bool = True,
    short_list_len: int = 16,
    long_list_len: int = 256,
    max_attempts: int = 4,
    short_list_multiplier: float = 0.5,
    auto_match_threshold: float = 0.75,
    use_best_auto_match: bool = False,
    candidate_scan_chunks: int = 5,
    template_path: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Merge`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = MergeConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        n_runs=n_runs,
        use_dummy=use_dummy,
        additional_instructions=additional_instructions,
        use_embeddings=use_embeddings,
        short_list_len=short_list_len,
        long_list_len=long_list_len,
        max_attempts=max_attempts,
        short_list_multiplier=short_list_multiplier,
        auto_match_threshold=auto_match_threshold,
        use_best_auto_match=use_best_auto_match,
        candidate_scan_chunks=candidate_scan_chunks,
        **cfg_kwargs,
    )
    return await Merge(cfg, template_path=template_path).run(
        df_left,
        df_right,
        on=on,
        left_on=left_on,
        right_on=right_on,
        how=how,
        reset_files=reset_files,
    )


async def filter(
    df: pd.DataFrame,
    column_name: str,
    *,
    condition: str,
    save_dir: str,
    entities_per_call: int = 150,
    shuffle: bool = True,
    random_seed: int = 42,
    n_runs: int = 1,
    threshold: float = 0.5,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-nano",
    n_parallels: int = 750,
    reset_files: bool = False,
    use_dummy: bool = False,
    file_name: str = "filter_responses.csv",
    max_timeout: Optional[float] = None,
    template_path: Optional[str] = None,
    **cfg_kwargs,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Filter`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = FilterConfig(
        condition=condition,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        entities_per_call=entities_per_call,
        shuffle=shuffle,
        random_seed=random_seed,
        n_runs=n_runs,
        threshold=threshold,
        additional_instructions=additional_instructions or "",
        use_dummy=use_dummy,
        max_timeout=max_timeout,
        **cfg_kwargs,
    )
    return await Filter(cfg, template_path=template_path).run(
        df,
        column_name,
        reset_files=reset_files,
    )


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


def view_coded_passages(
    df: pd.DataFrame,
    column_name: str,
    categories: Optional[Union[list[str], str]] = None,
    colab: bool = False,
):
    """Convenience wrapper for the passage viewer utility."""
    return _view_coded_passages(df, column_name, categories, colab=colab)
