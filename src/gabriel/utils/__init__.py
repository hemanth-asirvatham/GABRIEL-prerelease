"""Utility helpers for GABRIEL."""

from .openai_utils import get_response, get_all_responses
from .image_utils import encode_image
from .audio_utils import encode_audio
from .logging import get_logger, set_log_level
from .teleprompter import Teleprompter
from .mapmaker import MapMaker, create_county_choropleth
from .prompt_paraphraser import PromptParaphraser, PromptParaphraserConfig
from .parsing import safe_json, safest_json, clean_json_df
from .jinja import shuffled, shuffled_dict, get_env
from .passage_viewer import PassageViewer, view_coded_passages
from .word_matching import (
    normalize_text_aggressive,
    letters_only,
    robust_find_improved,
    strict_find,
)

__all__ = [
    "get_response",
    "get_all_responses",
    "get_logger",
    "set_log_level",
    "Teleprompter",
    "MapMaker",
    "create_county_choropleth",
    "PromptParaphraser",
    "PromptParaphraserConfig",
    "safe_json",
    "safest_json",
    "clean_json_df",
    "encode_image",
    "encode_audio",
    "shuffled",
    "shuffled_dict",
    "get_env",
    "normalize_text_aggressive",
    "letters_only",
    "robust_find_improved",
    "strict_find",
    "PassageViewer",
    "view_coded_passages",
]
