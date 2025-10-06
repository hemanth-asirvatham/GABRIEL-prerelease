import os
import pandas as pd
from typing import Callable, Dict, Optional, Union, Any, List

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
    Seed,
    SeedConfig,
    Filter,
    FilterConfig,
    Whatever,
    WhateverConfig,
    Ideate,
    IdeateConfig,
)
from .utils.openai_utils import get_all_responses
from .utils.passage_viewer import view_coded_passages as _view_coded_passages
from .tasks.debias import (
    DebiasConfig,
    DebiasPipeline,
    DebiasResult,
    MeasurementMode,
    RemovalMethod,
)

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


async def seed(
    instructions: str,
    *,
    save_dir: str,
    file_name: str = "seed_entities.csv",
    model: str = "o3-mini",
    n_parallels: int = 400,
    num_entities: int = 1000,
    entities_per_generation: int = 50,
    entity_batch_frac: float = 0.2,
    existing_entities_cap: int = 100,
    existing_sample_ratio: float = 0.5,
    use_dummy: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    max_timeout: Optional[float] = None,
    template_path: Optional[str] = None,
    existing_entities: Optional[List[str]] = None,
    reset_files: bool = False,
    **response_kwargs: Any,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Seed`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    cfg = SeedConfig(
        instructions=instructions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        n_parallels=n_parallels,
        num_entities=num_entities,
        entities_per_generation=entities_per_generation,
        entity_batch_frac=entity_batch_frac,
        existing_entities_cap=existing_entities_cap,
        existing_sample_ratio=existing_sample_ratio,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
        max_timeout=max_timeout,
    )
    task = Seed(cfg, template_path=template_path)
    return await task.run(
        existing_entities=existing_entities,
        reset_files=reset_files,
        **response_kwargs,
    )


async def classify(
    df: pd.DataFrame,
    column_name: Optional[str] = None,
    *,
    labels: dict[str, str],
    save_dir: str,
    additional_instructions: Optional[str] = None,
    model: str = "gpt-5-mini",
    differentiate: bool = False,
    circle_column_name: Optional[str] = None,
    square_column_name: Optional[str] = None,
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
        differentiate=differentiate,
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
        circle_column_name=circle_column_name,
        square_column_name=square_column_name,
        reset_files=reset_files,
    )


async def ideate(
    topic: str,
    *,
    save_dir: str,
    file_name: str = "ideation.csv",
    model: str = "gpt-5-mini",
    ranking_model: Optional[str] = None,
    n_ideas: int = 1000,
    n_parallels: int = 750,
    evaluation_mode: str = "recursive_rank",
    attributes: Optional[Dict[str, str]] = None,
    rank_attribute: Optional[str] = None,
    recursive_fraction: float = 1.0 / 3.0,
    recursive_min_remaining: int = 30,
    recursive_final_round_multiplier: int = 3,
    recursive_cut_side: str = "top",
    recursive_rate_first_round: bool = True,
    additional_instructions: Optional[str] = None,
    web_search: bool = False,
    use_dummy: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    reset_files: bool = False,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    rank_config_updates: Optional[Dict[str, Any]] = None,
    rank_run_kwargs: Optional[Dict[str, Any]] = None,
    rate_config_updates: Optional[Dict[str, Any]] = None,
    rate_run_kwargs: Optional[Dict[str, Any]] = None,
    use_seed_entities: Optional[bool] = None,
    seed_config_updates: Optional[Dict[str, Any]] = None,
    seed_run_kwargs: Optional[Dict[str, Any]] = None,
    template_path: Optional[str] = None,
) -> pd.DataFrame:
    """Convenience wrapper for :class:`gabriel.tasks.Ideate`."""

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    cfg_kwargs: Dict[str, Any] = dict(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        ranking_model=ranking_model,
        n_parallels=n_parallels,
        n_ideas=n_ideas,
        evaluation_mode=evaluation_mode,
        rank_attribute=rank_attribute,
        recursive_fraction=recursive_fraction,
        recursive_min_remaining=recursive_min_remaining,
        recursive_final_round_multiplier=recursive_final_round_multiplier,
        recursive_cut_side=recursive_cut_side,
        recursive_rate_first_round=recursive_rate_first_round,
        additional_instructions=additional_instructions,
        web_search=web_search,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )
    if attributes is not None:
        cfg_kwargs["attributes"] = attributes
    cfg = IdeateConfig(**cfg_kwargs)

    ideator = Ideate(cfg, template_path=template_path)
    return await ideator.run(
        topic,
        additional_instructions=additional_instructions,
        evaluation_mode=evaluation_mode,
        attributes=attributes,
        rank_attribute=rank_attribute,
        reset_files=reset_files,
        generation_kwargs=generation_kwargs,
        rank_config_updates=rank_config_updates,
        rank_run_kwargs=rank_run_kwargs,
        rate_config_updates=rate_config_updates,
        rate_run_kwargs=rate_run_kwargs,
        use_seed_entities=use_seed_entities,
        seed_config_updates=seed_config_updates,
        seed_run_kwargs=seed_run_kwargs,
    )


async def deidentify(
    df: pd.DataFrame,
    column_name: str,
    *,
    save_dir: str,
    grouping_column: Optional[str] = None,
    mapping_column: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    use_dummy: bool = False,
    file_name: str = "deidentified.csv",
    max_words_per_call: int = 7500,
    guidelines: str = "",
    additional_guidelines: str = "",
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    n_passes: int = 1,
    use_existing_mappings_only: bool = False,
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
        n_passes=n_passes,
        use_existing_mappings_only=use_existing_mappings_only,
        **cfg_kwargs,
    )
    return await Deidentifier(cfg, template_path=template_path).run(
        df,
        column_name,
        grouping_column=grouping_column,
        mapping_column=mapping_column,
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
    recursive: bool = False,
    recursive_fraction: float = 1.0 / 3.0,
    recursive_min_remaining: int = 30,
    recursive_final_round_multiplier: int = 3,
    recursive_cut_attr: Optional[str] = None,
    recursive_cut_side: str = "top",
    recursive_rate_first_round: bool = False,
    recursive_rewrite_func: Optional[Callable[[str, str, int], str]] = None,
    recursive_rewrite_text_col: str = "text",
    recursive_keep_stage_columns: bool = True,
    recursive_add_stage_suffix: bool = True,
    initial_rating_pass: bool = False,
    rate_kwargs: Optional[Dict[str, Any]] = None,
    id_column: Optional[str] = None,
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
        recursive=recursive,
        recursive_fraction=recursive_fraction,
        recursive_min_remaining=recursive_min_remaining,
        recursive_final_round_multiplier=recursive_final_round_multiplier,
        recursive_cut_attr=recursive_cut_attr,
        recursive_cut_side=recursive_cut_side,
        recursive_rate_first_round=recursive_rate_first_round,
        recursive_rewrite_func=recursive_rewrite_func,
        recursive_rewrite_text_col=recursive_rewrite_text_col,
        recursive_keep_stage_columns=recursive_keep_stage_columns,
        recursive_add_stage_suffix=recursive_add_stage_suffix,
        initial_rating_pass=initial_rating_pass,
        rate_kwargs=rate_kwargs or {},
        **cfg_kwargs,
    )
    return await Rank(cfg, template_path=template_path).run(
        df,
        column_name,
        id_column=id_column,
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
    modality: str = "text",
    json_mode: bool = True,
    max_timeout: Optional[float] = None,
    completion_check: bool = True,
    completion_max_rounds: int = 2,
    completion_classifier_instructions: Optional[str] = None,
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
        modality=modality,
        json_mode=json_mode,
        max_timeout=max_timeout,
        completion_check=completion_check,
        completion_max_rounds=completion_max_rounds,
        completion_classifier_instructions=completion_classifier_instructions,
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
    web_search: Optional[bool] = None,
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
    if "use_web_search" in cfg_kwargs and "web_search" not in cfg_kwargs:
        cfg_kwargs["web_search"] = cfg_kwargs.pop("use_web_search")
    else:
        cfg_kwargs.pop("use_web_search", None)

    cfg = ParaphraseConfig(
        instructions=instructions,
        revised_column_name=revised_column_name,
        n_revisions=n_revisions,
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        web_search=web_search,
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
    column_name: str,
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
        column_name=column_name,
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


async def debias(
    df: pd.DataFrame,
    column_name: str,
    *,
    mode: MeasurementMode = "rate",
    measurement_attribute: Optional[str] = None,
    removal_attribute: Optional[str] = None,
    signal_dictionary: Dict[str, str],
    attributes: Optional[Dict[str, str]] = None,
    removal_method: RemovalMethod = "codify",
    save_dir: str = os.path.expanduser("~/Documents/runs"),
    run_name: Optional[str] = None,
    strip_percentages: Optional[List[int]] = None,
    categories_to_strip: Optional[List[str]] = None,
    template_path: Optional[str] = None,
    model: str = "gpt-5-mini",
    n_parallels: int = 750,
    measurement_kwargs: Optional[Dict[str, Any]] = None,
    removal_kwargs: Optional[Dict[str, Any]] = None,
    max_words_per_call: Optional[int] = 1000,
    completion_max_rounds: Optional[int] = 3,
    use_dummy: bool = False,
    robust_regression: bool = True,
    random_seed: int = 12345,
    verbose: bool = True,
) -> DebiasResult:
    """Run the econometric debiasing pipeline on ``df[column_name]``.

    Parameters
    ----------
    measurement_attribute, removal_attribute:
        Specify the attribute used for regression and the key from
        ``signal_dictionary`` that should be removed.  When
        ``measurement_attribute`` is omitted the first key from
        ``attributes`` is used.  ``removal_attribute`` defaults to the
        measurement attribute when it exists in ``signal_dictionary`` or
        otherwise the first key from ``signal_dictionary``.  When defaults are
        inferred a notice is printed if ``verbose`` is ``True``.
    max_words_per_call, completion_max_rounds:
        Convenience passthroughs for the removal stage.  ``max_words_per_call``
        configures the codify task's chunk size, while
        ``completion_max_rounds`` is forwarded to codify and to the underlying
        paraphrase API calls.
    """

    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    measurement_kwargs = dict(measurement_kwargs or {})
    removal_kwargs = dict(removal_kwargs or {})

    if removal_method == "codify" and max_words_per_call is not None:
        removal_kwargs.setdefault("max_words_per_call", max_words_per_call)
    if completion_max_rounds is not None:
        removal_kwargs.setdefault("completion_max_rounds", completion_max_rounds)

    cfg = DebiasConfig(
        mode=mode,
        measurement_attribute=measurement_attribute,
        removal_attribute=removal_attribute,
        signal_dictionary=signal_dictionary,
        attributes=attributes or {},
        removal_method=removal_method,
        save_dir=save_dir,
        run_name=run_name,
        strip_percentages=strip_percentages,
        categories_to_strip=categories_to_strip,
        template_path=template_path,
        model=model,
        n_parallels=n_parallels,
        measurement_kwargs=measurement_kwargs,
        removal_kwargs=removal_kwargs,
        use_dummy=use_dummy,
        robust_regression=robust_regression,
        random_seed=random_seed,
        verbose=verbose,
    )
    pipeline = DebiasPipeline(cfg)
    return await pipeline.run(df, column_name)


async def whatever(
    prompts: Union[str, list[str], pd.DataFrame],
    identifiers: Optional[list[str]] = None,
    *,
    save_dir: str,
    column_name: Optional[str] = None,
    identifier_column: Optional[str] = None,
    image_column: Optional[str] = None,
    audio_column: Optional[str] = None,
    prompt_images: Optional[Dict[str, List[str]]] = None,
    prompt_audio: Optional[Dict[str, List[Dict[str, str]]]] = None,
    file_name: str = "custom_prompt_responses.csv",
    model: str = "gpt-5-mini",
    json_mode: bool = False,
    web_search: Optional[bool] = None,
    web_search_filters: Optional[Dict[str, Any]] = None,
    search_context_size: str = "medium",
    n_parallels: int = 750,
    use_dummy: bool = False,
    reset_files: bool = False,
    reasoning_effort: Optional[str] = None,
    reasoning_summary: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """Wrapper around :func:`get_all_responses` for arbitrary prompts.

    Results are saved to ``save_dir/file_name``.  Web search features can be
    customised with ``web_search_filters`` and ``search_context_size`` (both of
    which map directly to :func:`gabriel.utils.openai_utils.get_all_responses`).
    Filters accept ``allowed_domains`` plus optional location hints
    (``city``, ``country``, ``region``, ``timezone`` and ``type`` – typically
    ``"approximate"``) that are forwarded to the OpenAI Responses API.
    The ``web_search`` flag mirrors the OpenAI Python client.  Passing the
    legacy ``use_web_search`` keyword is still supported and will be coerced to
    ``web_search`` automatically.
    """
    save_dir = os.path.expandvars(os.path.expanduser(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    if web_search is None and "web_search" in kwargs:
        web_search = kwargs.pop("web_search")
    else:
        kwargs.pop("web_search", None)

    legacy_use_web_search = kwargs.pop("use_web_search", None)
    if web_search is None and legacy_use_web_search is not None:
        web_search = bool(legacy_use_web_search)

    if web_search_filters is None and "web_search_filters" in kwargs:
        web_search_filters = kwargs.pop("web_search_filters")
    else:
        kwargs.pop("web_search_filters", None)

    if "search_context_size" in kwargs:
        if search_context_size == "medium":
            search_context_size = kwargs.pop("search_context_size")
        else:
            kwargs.pop("search_context_size")

    cfg = WhateverConfig(
        save_dir=save_dir,
        file_name=file_name,
        model=model,
        json_mode=json_mode,
        web_search=web_search,
        web_search_filters=web_search_filters,
        search_context_size=search_context_size,
        n_parallels=n_parallels,
        use_dummy=use_dummy,
        reasoning_effort=reasoning_effort,
        reasoning_summary=reasoning_summary,
    )

    runner = Whatever(cfg)
    return await runner.run(
        prompts,
        identifiers=identifiers,
        column_name=column_name,
        identifier_column=identifier_column,
        image_column=image_column,
        audio_column=audio_column,
        prompt_images=prompt_images,
        prompt_audio=prompt_audio,
        web_search_filters=web_search_filters,
        reset_files=reset_files,
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
