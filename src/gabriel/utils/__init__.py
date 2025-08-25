"""Utility helpers for GABRIEL."""

from .openai_utils import (
    get_response,
    get_all_responses,
    get_embedding,
    get_all_embeddings,
)
from .image_utils import encode_image
from .audio_utils import encode_audio
from .media_utils import load_image_inputs, load_audio_inputs
from .logging import get_logger, set_log_level
from .mapmaker import MapMaker, create_county_choropleth
from .parsing import safe_json, safest_json, clean_json_df
from .jinja import shuffled, shuffled_dict, get_env
from .passage_viewer import PassageViewer, view_coded_passages
from .word_matching import (
    normalize_text_aggressive,
    letters_only,
    robust_find_improved,
    strict_find,
)
from .prompt_utils import swap_circle_square
from .file_utils import load_files

__all__ = [
    "get_response",
    "get_all_responses",
    "get_embedding",
    "get_all_embeddings",
    "get_logger",
    "set_log_level",
    "MapMaker",
    "create_county_choropleth",
    "safe_json",
    "safest_json",
    "clean_json_df",
    "encode_image",
    "encode_audio",
    "load_image_inputs",
    "load_audio_inputs",
    "shuffled",
    "shuffled_dict",
    "get_env",
    "normalize_text_aggressive",
    "letters_only",
    "robust_find_improved",
    "strict_find",
    "PassageViewer",
    "view_coded_passages",
    "swap_circle_square",
    "load_files",
]
