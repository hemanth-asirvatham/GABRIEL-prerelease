from __future__ import annotations

import re
from typing import Dict

_WORD_MAP: Dict[str, str] = {
    "circle": "square",
    "square": "circle",
    "Circle": "Square",
    "Square": "Circle",
    "CIRCLE": "SQUARE",
    "SQUARE": "CIRCLE",
}


def swap_circle_square(text: str) -> str:
    """Swap occurrences of 'circle' and 'square' in ``text`` while preserving case.

    The function temporarily masks Jinja variable names like ``entry_circle`` and
    ``entry_square`` to avoid mangling them during the swap.
    """

    placeholders = {
        "{{ entry_circle }}": "__ENTRY_CIRCLE__",
        "{{ entry_square }}": "__ENTRY_SQUARE__",
        "entry_circle": "__ENTRY_CIRCLE_VAR__",
        "entry_square": "__ENTRY_SQUARE_VAR__",
    }

    for key, token in placeholders.items():
        text = text.replace(key, token)

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        return _WORD_MAP.get(word, word)

    text = re.sub(r"\b(circle|square|Circle|Square|CIRCLE|SQUARE)\b", repl, text)

    for key, token in placeholders.items():
        text = text.replace(token, key)

    return text
