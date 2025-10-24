from __future__ import annotations

import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, InitVar
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ..core.prompt_template import PromptTemplate, resolve_template
from ..tasks.classify import Classify, ClassifyConfig
from ..utils import (
    get_all_responses,
    letters_only,
    load_audio_inputs,
    load_image_inputs,
    normalize_text_aggressive,
    robust_find_improved,
    safe_json,
    strict_find,
)


@dataclass
class CodifyConfig:
    """Configuration for :class:`Codify`."""

    save_dir: str
    file_name: str = "coding_results.csv"
    model: str = "gpt-5-mini"
    n_parallels: int = 750
    max_words_per_call: int = 1000
    max_categories_per_call: int = 8
    debug_print: bool = False
    use_dummy: bool = False
    reasoning_effort: Optional[str] = None
    reasoning_summary: Optional[str] = None
    modality: str = "text"
    json_mode: bool = True
    max_timeout: Optional[float] = None
    n_rounds: int = 2  # Total Codify passes including the initial run; set to 1 to skip completion sweeps
    completion_classifier_instructions: Optional[str] = None
    completion_max_rounds: InitVar[Optional[int]] = None

    def __post_init__(self, completion_max_rounds: Optional[int]) -> None:
        if completion_max_rounds is not None:
            warnings.warn(
                "completion_max_rounds is deprecated; use n_rounds instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.n_rounds = completion_max_rounds

        try:
            rounds = int(self.n_rounds)
        except (TypeError, ValueError):
            rounds = 1
        if rounds < 1:
            rounds = 1
        self.n_rounds = rounds


@dataclass
class ChunkResult:
    """Container holding the parsed response for a single chunk."""

    identifier: str
    chunk_text: str
    data: Dict[str, Any]


@dataclass
class PromptRequest:
    """Metadata for a prompt dispatched to the model."""

    identifier: str
    prompt: str
    row_index: int
    chunk_text: str


class Codify:
    """Pipeline for coding passages of text according to specified categories."""

    def __init__(
        self,
        cfg: CodifyConfig,
        template: Optional[PromptTemplate] = None,
        template_path: Optional[str] = None,
    ) -> None:
        """Create a new :class:`Codify` instance.

        Parameters
        ----------
        template:
            Optional preconstructed :class:`PromptTemplate`.
        template_path:
            Path to a custom Jinja2 template on disk.  The template is
            validated to ensure it exposes the same variables as the
            built-in ``codify_prompt.jinja2`` template.
        """
    
        expanded = Path(os.path.expandvars(os.path.expanduser(cfg.save_dir)))
        expanded.mkdir(parents=True, exist_ok=True)
        cfg.save_dir = str(expanded)
        self.cfg = cfg
        self.hit_rate_stats = {}  # Track hit rates across all texts
        self.template = resolve_template(
            template=template,
            template_path=template_path,
            reference_filename="codify_prompt.jinja2",
        )
        self.hit_rate_stats = {}  # Track hit rates across all texts

    @staticmethod
    def view(
        df: pd.DataFrame,
        column_name: str,
        attributes: Optional[Union[List[str], str]] = None,
        **viewer_kwargs: Any,
    ):
        """Convenience wrapper around :func:`gabriel.view`.

        This helper makes it easy to visualise coding results produced by
        :class:`Codify`. Additional keyword arguments are forwarded to
        :func:`gabriel.view`, enabling features such as the Colab viewer,
        attribute chips, and custom metadata headers.
        """

        from ..utils import view as view_results

        normalized_attributes = attributes or "coded_passages"

        return view_results(
            df,
            column_name,
            attributes=normalized_attributes,
            **viewer_kwargs,
        )

    def parse_json(self, response_text: Any) -> Optional[dict]:
        """Robust JSON parsing using :func:`safe_json`."""

        parsed = safe_json(response_text)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            inner = safe_json(parsed[0])
            if isinstance(inner, dict):
                return inner
        return None

    def chunk_by_words(self, text: str, max_words: int) -> List[str]:
        """Split text into chunks by word count."""
        words = text.split()
        if len(words) <= max_words:
            return [text]
        return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]

    def find_snippet_in_text(self, text: str, beginning_excerpt: str, ending_excerpt: str) -> Optional[str]:
        """Fast snippet finding that returns actual text from the original document."""
        if not beginning_excerpt:
            return None
        
        # Handle short excerpts (no ending)
        if not ending_excerpt:
            match = robust_find_improved(text, beginning_excerpt)
            if match:
                # Find the actual position in the original text
                start_pos, end_pos, match_type = self._find_actual_position_with_type(text, beginning_excerpt)
                if start_pos is not None:
                    # Always ensure word boundaries and add some context
                    start_pos = self._find_word_start(text, start_pos)
                    end_pos = self._find_word_end(text, end_pos)
                    
                    # If using fallback matching, expand to include more context
                    if match_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                        start_pos, end_pos = self._expand_fallback_match(text, start_pos, end_pos, beginning_excerpt)
                    else:
                        # Add minimal context for exact matches
                        words_after = self._get_n_words_after(text, end_pos, 5)
                        end_pos = min(len(text), end_pos + len(words_after))
                    
                    return text[start_pos:end_pos].strip()
            return None
        
        # Handle longer snippets with both beginning and ending
        begin_match = robust_find_improved(text, beginning_excerpt)
        end_match = robust_find_improved(text, ending_excerpt)
        
        if not begin_match and not end_match:
            return None
        elif begin_match and not end_match:
            # Beginning found but ending not found - include beginning + 20 words after
            begin_start, begin_end, _ = self._find_actual_position_with_type(text, beginning_excerpt)
            if begin_start is not None:
                # Find word boundary at beginning end
                word_end = self._find_word_end(text, begin_end)
                words_after = self._get_n_words_after(text, word_end, 20)
                
                # Calculate end position for the 20 words after
                after_end = min(len(text), word_end + len(words_after))
                result = text[begin_start:after_end].strip()
                return result if result else None
            return None
        elif not begin_match and end_match:
            # Ending found but beginning not found - include 20 words before + ending
            end_start, end_end, _ = self._find_actual_position_with_type(text, ending_excerpt)
            if end_start is not None:
                # Find word boundary at ending start
                word_start = self._find_word_start(text, end_start)
                words_before = self._get_n_words_before(text, word_start, 20)
                
                # Calculate start position for the 20 words before
                before_start = max(0, word_start - len(words_before))
                result = text[before_start:end_end].strip()
                return result if result else None
            return None
        else:
            # Both beginning and ending found - extract the actual snippet from original text
            begin_start, begin_end, begin_type = self._find_actual_position_with_type(text, beginning_excerpt)
            end_start, end_end, end_type = self._find_actual_position_with_type(text, ending_excerpt)
            
            if begin_start is not None and end_start is not None:
                # Expand fallback matches
                if begin_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                    begin_start, begin_end = self._expand_fallback_match(text, begin_start, begin_end, beginning_excerpt)
                if end_type in ['first_20', 'last_20', 'first_half', 'second_half']:
                    end_start, end_end = self._expand_fallback_match(text, end_start, end_end, ending_excerpt)
                
                # Make sure ending comes after beginning
                if end_start >= begin_start:
                    return text[begin_start:end_end].strip()
                else:
                    # Ending comes before beginning, just return beginning snippet
                    return self.find_snippet_in_text(text, beginning_excerpt, "")
            elif begin_start is not None:
                # Only beginning found
                return self.find_snippet_in_text(text, beginning_excerpt, "")
            
            return None

    def _find_actual_position(self, text: str, excerpt: str, _recursion_depth: int = 0) -> tuple:
        """Find the actual character positions of an excerpt in the original text."""
        result = self._find_actual_position_with_type(text, excerpt, _recursion_depth)
        return result[0], result[1]  # Return just position, not match type

    def _find_actual_position_with_type(self, text: str, excerpt: str, _recursion_depth: int = 0) -> tuple:
        """Find the actual character positions and match type using the SAME permissive strategies as robust_find_improved."""
        if not excerpt.strip():
            return None, None, None
        
        # Prevent infinite recursion
        if _recursion_depth > 1:
            return None, None, None
        
        # Strategy 1: Try direct matching first (fastest)
        text_lower = text.lower()
        excerpt_lower = excerpt.lower().strip()
        idx = text_lower.find(excerpt_lower)
        if idx != -1:
            return idx, idx + len(excerpt_lower), 'exact'
        
        # Strategy 2: Try with our aggressive normalization
        text_norm = normalize_text_aggressive(text)
        excerpt_norm = normalize_text_aggressive(excerpt)
        
        idx = text_norm.lower().find(excerpt_norm.lower())
        if idx != -1:
            # Map back to original text position approximately
            start_pos, end_pos = self._map_normalized_to_original(text, text_norm, idx, len(excerpt_norm))
            return start_pos, end_pos, 'normalized'
        
        # Strategy 3: Letters-only matching (same as robust_find_improved)
        text_letters = letters_only(text)
        excerpt_letters = letters_only(excerpt)
        
        if excerpt_letters and excerpt_letters in text_letters:
            letters_idx = text_letters.find(excerpt_letters)
            ratio = letters_idx / len(text_letters) if text_letters else 0
            approx_start = int(ratio * len(text))
            return approx_start, approx_start + len(excerpt), 'letters_only'
        
        # Strategy 4: First 20 characters fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_first_20 = excerpt_letters[:20]
            if excerpt_first_20 in text_letters:
                letters_idx = text_letters.find(excerpt_first_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'first_20'
        
        # Strategy 5: Last 20 characters fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_last_20 = excerpt_letters[-20:]
            if excerpt_last_20 in text_letters:
                letters_idx = text_letters.find(excerpt_last_20)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'last_20'
        
        # Strategy 6: First + Last 10 fallback (same as robust_find_improved)
        if len(excerpt_letters) >= 20:
            excerpt_first_10 = excerpt_letters[:10]
            excerpt_last_10 = excerpt_letters[-10:]
            if excerpt_first_10 in text_letters and excerpt_last_10 in text_letters:
                letters_idx = text_letters.find(excerpt_first_10)
                ratio = letters_idx / len(text_letters) if text_letters else 0
                approx_start = int(ratio * len(text))
                return approx_start, approx_start + len(excerpt), 'first_last_10'
        
        # Strategy 7: Half matching for shorter excerpts (same as robust_find_improved)
        if 10 <= len(excerpt_letters) < 20:
            excerpt_first_half = excerpt_letters[:len(excerpt_letters)//2]
            excerpt_second_half = excerpt_letters[len(excerpt_letters)//2:]
            if len(excerpt_first_half) >= 5 and len(excerpt_second_half) >= 5:
                if excerpt_first_half in text_letters and excerpt_second_half in text_letters:
                    letters_idx = text_letters.find(excerpt_first_half)
                    ratio = letters_idx / len(text_letters) if text_letters else 0
                    approx_start = int(ratio * len(text))
                    return approx_start, approx_start + len(excerpt), 'first_half'
        
        return None, None, None

    def _map_normalized_to_original(self, original: str, normalized: str, norm_start: int, norm_length: int) -> tuple:
        """Map a position in normalized text back to original text."""
        # This is an approximation - we'll search around the estimated area
        if len(normalized) == 0:
            return None, None
        
        # Estimate the ratio
        ratio_start = norm_start / len(normalized)
        ratio_end = (norm_start + norm_length) / len(normalized)
        
        # Estimate positions in original text
        orig_start_est = int(ratio_start * len(original))
        orig_end_est = int(ratio_end * len(original))
        
        # Expand search window
        window_size = max(50, norm_length * 2)
        search_start = max(0, orig_start_est - window_size)
        search_end = min(len(original), orig_end_est + window_size)
        
        # Try to find the best match in this window
        search_text = original[search_start:search_end]
        excerpt_to_find = normalized[norm_start:norm_start + norm_length]
        
        # Simple substring search in the window
        for i in range(len(search_text) - len(excerpt_to_find) + 1):
            window = search_text[i:i + len(excerpt_to_find)]
            if normalize_text_aggressive(window).lower() == excerpt_to_find.lower():
                return search_start + i, search_start + i + len(window)
        
        # Fallback: return estimated positions
        return max(0, orig_start_est), min(len(original), orig_end_est)

    def _expand_fallback_match(self, text: str, start_pos: int, end_pos: int, original_excerpt: str) -> tuple:
        """Expand a fallback match to include proper word boundaries and context."""
        # Find word boundaries around the match
        new_start = self._find_word_start(text, start_pos)
        new_end = self._find_word_end(text, end_pos)
        
        # Add some context words for better snippet quality
        words_before = self._get_n_words_before(text, new_start, 3)
        words_after = self._get_n_words_after(text, new_end, 3)
        
        # Calculate final boundaries
        final_start = max(0, new_start - len(words_before))
        final_end = min(len(text), new_end + len(words_after))
        
        return final_start, final_end

    def _find_word_start(self, text: str, pos: int) -> int:
        """Find the start of the word containing the given position."""
        if pos <= 0:
            return 0
        # Move backwards to find word boundary
        while pos > 0 and text[pos-1].isalnum():
            pos -= 1
        return pos

    def _find_word_end(self, text: str, pos: int) -> int:
        """Find the end of the word containing the given position."""
        if pos >= len(text):
            return len(text)
        # Move forwards to find word boundary
        while pos < len(text) and text[pos].isalnum():
            pos += 1
        return pos

    def _get_n_words_before(self, text: str, pos: int, n: int) -> str:
        """Get n words before the given position."""
        if pos <= 0:
            return ""
        
        # Look backwards from position to find word boundaries
        before_text = text[:pos]
        words = before_text.split()
        
        if len(words) <= n:
            return before_text
        else:
            return " ".join(words[-n:]) + " "

    def _get_n_words_after(self, text: str, pos: int, n: int) -> str:
        """Get n words after the given position."""
        if pos >= len(text):
            return ""
        
        # Look forwards from position to find word boundaries
        after_text = text[pos:]
        words = after_text.split()
        
        if len(words) <= n:
            return after_text
        else:
            return " " + " ".join(words[:n])

    def consolidate_snippets(
        self,
        original_text: str,
        chunk_results: List[ChunkResult],
        category: str,
        *,
        debug_print: bool = False,
    ) -> List[str]:
        """Convert per-chunk responses into verbatim snippets for ``category``."""

        all_excerpts: List[Tuple[str, str]] = []
        chunk_texts: List[str] = []
        for chunk_result in chunk_results:
            payload = chunk_result.data
            if not isinstance(payload, dict):
                continue
            if category in payload and isinstance(payload[category], list):
                for item in payload[category]:
                    if isinstance(item, dict):
                        beginning = item.get("beginning excerpt", "")
                        ending = item.get("ending excerpt", "")
                        if beginning:
                            all_excerpts.append((beginning, ending))
                            chunk_texts.append(chunk_result.chunk_text)

        found = 0
        snippets: List[str] = []
        failed: List[Tuple[str, str]] = []
        begin_fail_count = 0
        end_fail_count = 0
        strict_matches = 0

        for idx, (beginning, ending) in enumerate(all_excerpts):
            chunk_text = chunk_texts[idx] if idx < len(chunk_texts) else ""
            snippet: Optional[str] = None

            strict_begin = strict_find(original_text, beginning)
            strict_end = strict_find(original_text, ending) if ending and ending.strip() else True
            if strict_begin and strict_end:
                strict_matches += 1

            if chunk_text:
                snippet = self.find_snippet_in_text(chunk_text, beginning, ending)
                if debug_print and snippet:
                    print(f"[DEBUG] Found in chunk: '{beginning[:50]}...'")

            if not snippet:
                snippet = self.find_snippet_in_text(original_text, beginning, ending)
                if debug_print and snippet:
                    print(f"[DEBUG] Found in full text: '{beginning[:50]}...'")
                elif debug_print:
                    print(f"[DEBUG] FAILED to find: '{beginning[:50]}...'")
                    letters_begin = letters_only(beginning)
                    letters_text = letters_only(original_text)
                    print(f"[DEBUG] Letters-only excerpt: '{letters_begin[:50]}...'")
                    print(f"[DEBUG] Letters-only contains: {letters_begin in letters_text}")
                    begin_match = robust_find_improved(original_text, beginning)
                    end_match = robust_find_improved(original_text, ending) if ending else True
                    print(f"[DEBUG] Failure analysis for '{beginning[:30]}...':")
                    print(f"[DEBUG]   Begin match: {begin_match is not None}")
                    print(
                        f"[DEBUG]   End match: {end_match is not None} (ending: '{ending[:20]}...' if ending else 'None')"
                    )

            if snippet:
                if snippet not in snippets:
                    snippets.append(snippet)
                    found += 1
            else:
                begin_match = strict_find(original_text, beginning)
                if ending and ending.strip():
                    end_match = strict_find(original_text, ending)
                    if not end_match:
                        end_fail_count += 1
                if not begin_match:
                    begin_fail_count += 1
                failed.append((beginning, ending))

        total = len(all_excerpts)
        if total:
            stats = self.hit_rate_stats.setdefault(
                category,
                {
                    "found": 0,
                    "total": 0,
                    "failed_examples": [],
                    "begin_failures": 0,
                    "end_failures": 0,
                    "strict_matches": 0,
                },
            )
            stats["found"] += found
            stats["total"] += total
            stats["begin_failures"] += begin_fail_count
            stats["end_failures"] += end_fail_count
            stats["strict_matches"] += strict_matches
            if failed and len(stats["failed_examples"]) < 3:
                stats["failed_examples"].extend(failed[:2])

        if debug_print and total:
            rate = 100.0 * found / total if total else 0.0
            strict_rate = 100.0 * strict_matches / total if total else 0.0
            print(
                f"[DEBUG] Category '{category}': {found}/{total} matched ({rate:.1f}%)"
                f" | Strict: {strict_matches} ({strict_rate:.1f}%)"
                f" | Begin failures: {begin_fail_count} | End failures: {end_fail_count}"
            )

        return snippets

    def print_final_hit_rates(self) -> None:
        """Print aggregated hit-rate statistics for debugging."""

        if not self.hit_rate_stats:
            return

        print("\n" + "=" * 80)
        print("FINAL MATCHING STATISTICS")
        print("=" * 80)

        total_found = 0
        total_excerpts = 0
        total_begin_failures = 0
        total_end_failures = 0
        total_strict_matches = 0

        for category in sorted(self.hit_rate_stats.keys()):
            stats = self.hit_rate_stats[category]
            found = stats.get("found", 0)
            total = stats.get("total", 0)
            begin_fail = stats.get("begin_failures", 0)
            end_fail = stats.get("end_failures", 0)
            strict_match = stats.get("strict_matches", 0)
            hit_rate = 100.0 * found / total if total else 0.0
            strict_rate = 100.0 * strict_match / total if total else 0.0
            begin_fail_pct = 100.0 * begin_fail / total if total else 0.0
            end_fail_pct = 100.0 * end_fail / total if total else 0.0

            print(
                f"{category:25s}: {found:3d}/{total:3d} ({hit_rate:4.1f}%) | "
                f"Strict: {strict_match:3d} ({strict_rate:4.1f}%) | "
                f"Begin fails: {begin_fail:2d} ({begin_fail_pct:4.1f}%) | "
                f"End fails: {end_fail:2d} ({end_fail_pct:4.1f}%)"
            )

            total_found += found
            total_excerpts += total
            total_begin_failures += begin_fail
            total_end_failures += end_fail
            total_strict_matches += strict_match

        overall_rate = 100.0 * total_found / total_excerpts if total_excerpts else 0.0
        overall_strict_rate = (
            100.0 * total_strict_matches / total_excerpts if total_excerpts else 0.0
        )
        overall_begin_fail_rate = (
            100.0 * total_begin_failures / total_excerpts if total_excerpts else 0.0
        )
        overall_end_fail_rate = (
            100.0 * total_end_failures / total_excerpts if total_excerpts else 0.0
        )

        print("-" * 80)
        print(
            f"{'OVERALL':25s}: {total_found:3d}/{total_excerpts:3d} ({overall_rate:4.1f}%) | "
            f"Strict: {total_strict_matches:3d} ({overall_strict_rate:4.1f}%) | "
            f"Begin fails: {total_begin_failures:2d} ({overall_begin_fail_rate:4.1f}%) | "
            f"End fails: {total_end_failures:2d} ({overall_end_fail_rate:4.1f}%)"
        )
        print("=" * 80)

    def _iteration_file_name(self, iteration: int) -> str:
        if iteration == 0:
            return self.cfg.file_name
        stem, ext = os.path.splitext(self.cfg.file_name)
        return f"{stem}_iter{iteration}{ext}"

    def _strip_snippets(self, text: str, snippets_by_category: Dict[str, List[str]]) -> str:
        remaining = text
        for snippets in snippets_by_category.values():
            for snippet in snippets:
                if snippet:
                    remaining = remaining.replace(snippet, " ", 1)
        return re.sub(r"\s+", " ", remaining).strip()

    def _merge_snippet_results(
        self,
        destination: Dict[int, Dict[str, List[str]]],
        source: Dict[int, Dict[str, List[str]]],
    ) -> bool:
        added = False
        for row_idx, cat_map in source.items():
            dest_row = destination.setdefault(row_idx, {})
            for category, snippets in cat_map.items():
                dest_list = dest_row.setdefault(category, [])
                for snippet in snippets:
                    if snippet and snippet not in dest_list:
                        dest_list.append(snippet)
                        added = True
        return added

    async def _gather_iteration(
        self,
        row_texts: Dict[int, str],
        *,
        original_texts: List[str],
        raw_values: List[Any],
        categories: Optional[Dict[str, str]],
        additional_instructions: Optional[str],
        iteration: int,
        dynamic_mode: bool,
        reset_files: bool,
        category_subset: Optional[Set[str]] = None,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, List[str]]]:
        if not row_texts:
            return {}

        debug = self.cfg.debug_print
        selected_categories: List[str] = []
        if not dynamic_mode and categories:
            selected_categories = [
                cat
                for cat in categories.keys()
                if category_subset is None or cat in category_subset
            ]
            if not selected_categories:
                return {}

        requests: List[PromptRequest] = []
        prompt_images: Dict[str, List[str]] = {}
        prompt_audio: Dict[str, List[Dict[str, str]]] = {}

        if not dynamic_mode and categories:
            category_batches = [
                selected_categories[i : i + self.cfg.max_categories_per_call]
                for i in range(0, len(selected_categories), self.cfg.max_categories_per_call)
            ]
        else:
            category_batches = []

        for row_idx, text in row_texts.items():
            text_str = str(text or "")
            if not text_str.strip():
                continue
            chunks = self.chunk_by_words(text_str, self.cfg.max_words_per_call)
            images = (
                load_image_inputs(raw_values[row_idx])
                if self.cfg.modality == "image"
                else None
            )
            audio_inputs = (
                load_audio_inputs(raw_values[row_idx])
                if self.cfg.modality == "audio"
                else None
            )

            for chunk_idx, chunk in enumerate(chunks):
                if dynamic_mode:
                    identifier = f"row{row_idx}_iter{iteration}_chunk{chunk_idx}"
                    prompt = self.template.render(
                        text=chunk,
                        categories=None,
                        additional_instructions=additional_instructions,
                        modality=self.cfg.modality,
                    )
                    requests.append(
                        PromptRequest(
                            identifier=identifier,
                            prompt=prompt,
                            row_index=row_idx,
                            chunk_text=chunk,
                        )
                    )
                    if images:
                        prompt_images[identifier] = list(images)
                    if audio_inputs:
                        prompt_audio[identifier] = list(audio_inputs)
                else:
                    for batch_idx, batch_keys in enumerate(category_batches):
                        assert categories is not None
                        batch_categories = {k: categories[k] for k in batch_keys}
                        identifier = (
                            f"row{row_idx}_iter{iteration}_chunk{chunk_idx}_batch{batch_idx}"
                        )
                        prompt = self.template.render(
                            text=chunk,
                            categories=batch_categories,
                            additional_instructions=additional_instructions,
                            modality=self.cfg.modality,
                        )
                        requests.append(
                            PromptRequest(
                                identifier=identifier,
                                prompt=prompt,
                                row_index=row_idx,
                                chunk_text=chunk,
                            )
                        )
                        if images:
                            prompt_images[identifier] = list(images)
                        if audio_inputs:
                            prompt_audio[identifier] = list(audio_inputs)

        if not requests:
            return {}

        prompts = [req.prompt for req in requests]
        identifiers = [req.identifier for req in requests]
        id_to_request = {req.identifier: req for req in requests}

        batch_df = await get_all_responses(
            prompts=prompts,
            identifiers=identifiers,
            n_parallels=self.cfg.n_parallels,
            save_path=os.path.join(self.cfg.save_dir, self._iteration_file_name(iteration)),
            reset_files=reset_files,
            use_dummy=self.cfg.use_dummy,
            json_mode=self.cfg.json_mode,
            model=self.cfg.model,
            max_timeout=self.cfg.max_timeout,
            print_example_prompt=True,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
            prompt_images=prompt_images or None,
            prompt_audio=prompt_audio or None,
            **kwargs,
        )

        chunk_results_by_row: Dict[int, List[ChunkResult]] = defaultdict(list)
        for ident, resp in zip(batch_df["Identifier"], batch_df["Response"]):
            request = id_to_request.get(ident)
            if request is None:
                continue
            main = resp[0] if isinstance(resp, list) and resp else resp
            parsed = self.parse_json(main) or {}
            if debug:
                if not parsed:
                    print(f"[DEBUG] Failed to parse response for {ident}")
                else:
                    print(f"[DEBUG] Parsed response for {ident} with keys: {list(parsed.keys())}")
            chunk_results_by_row[request.row_index].append(
                ChunkResult(identifier=ident, chunk_text=request.chunk_text, data=parsed)
            )

        iteration_results: Dict[int, Dict[str, List[str]]] = {}
        if dynamic_mode:
            for row_idx in row_texts.keys():
                chunk_results = chunk_results_by_row.get(row_idx, [])
                if not chunk_results:
                    continue
                categories_seen: Set[str] = set()
                for chunk_result in chunk_results:
                    for key, value in chunk_result.data.items():
                        if isinstance(key, str) and isinstance(value, list):
                            categories_seen.add(key)
                if not categories_seen:
                    continue
                row_map: Dict[str, List[str]] = {}
                for category in sorted(categories_seen):
                    snippets = self.consolidate_snippets(
                        original_texts[row_idx],
                        chunk_results,
                        category,
                        debug_print=debug,
                    )
                    if snippets:
                        row_map[category] = snippets
                if row_map:
                    iteration_results[row_idx] = row_map
        else:
            for row_idx in row_texts.keys():
                chunk_results = chunk_results_by_row.get(row_idx, [])
                row_map: Dict[str, List[str]] = {}
                for category in selected_categories:
                    snippets = self.consolidate_snippets(
                        original_texts[row_idx],
                        chunk_results,
                        category,
                        debug_print=debug,
                    )
                    row_map[category] = snippets
                iteration_results[row_idx] = row_map

        return iteration_results

    async def _classify_remaining(
        self,
        aggregated: Dict[int, Dict[str, List[str]]],
        original_texts: List[str],
        categories: Dict[str, str],
        additional_instructions: Optional[str],
        iteration: int,
        reset_files: bool,
    ) -> Dict[int, Set[str]]:
        row_indices: List[int] = []
        remaining_texts: List[str] = []
        for row_idx, original in enumerate(original_texts):
            snippet_map = aggregated.get(row_idx, {})
            stripped = self._strip_snippets(original, snippet_map)
            if stripped:
                row_indices.append(row_idx)
                remaining_texts.append(stripped)

        if not row_indices:
            return {}

        validation_dir = os.path.join(self.cfg.save_dir, "completion_checks")
        os.makedirs(validation_dir, exist_ok=True)
        stem = Path(self.cfg.file_name).stem
        file_name = f"{stem}_completion_iter{iteration}.csv"

        base_instruction = (
            "These passages contain the remaining text after previously extracted snippets were removed. "
            "Return True for a label only if the remaining text still contains a clear, distinct snippet "
            "that should be coded for that label. Default to False when unsure."
        )
        if self.cfg.completion_classifier_instructions:
            base_instruction += "\n" + self.cfg.completion_classifier_instructions.strip()
        if additional_instructions:
            base_instruction += "\nOriginal coding instructions:\n" + additional_instructions.strip()

        classify_cfg = ClassifyConfig(
            labels=categories,
            save_dir=validation_dir,
            file_name=file_name,
            model=self.cfg.model,
            n_parallels=self.cfg.n_parallels,
            n_runs=1,
            use_dummy=self.cfg.use_dummy,
            additional_instructions=base_instruction,
            modality=self.cfg.modality,
            n_attributes_per_run=self.cfg.max_categories_per_call,
            reasoning_effort=self.cfg.reasoning_effort,
            reasoning_summary=self.cfg.reasoning_summary,
        )
        classifier = Classify(classify_cfg)

        cls_df = pd.DataFrame({"text": remaining_texts})
        results_df = await classifier.run(
            cls_df,
            column_name="text",
            reset_files=reset_files and iteration == 0,
        )

        flagged: Dict[int, Set[str]] = {}
        for idx, row_idx in enumerate(row_indices):
            flagged_categories = {
                category
                for category in categories.keys()
                if bool(results_df.at[idx, category])
            }
            if flagged_categories:
                flagged[row_idx] = flagged_categories

        return flagged

    async def _completion_loop(
        self,
        aggregated: Dict[int, Dict[str, List[str]]],
        original_texts: List[str],
        raw_values: List[Any],
        categories: Dict[str, str],
        additional_instructions: Optional[str],
        reset_files: bool,
        **kwargs: Any,
    ) -> Dict[int, Dict[str, List[str]]]:
        total_rounds = max(1, int(self.cfg.n_rounds))
        completion_iterations = max(0, total_rounds - 1)

        for depth in range(1, completion_iterations + 1):
            flagged = await self._classify_remaining(
                aggregated,
                original_texts,
                categories,
                additional_instructions,
                iteration=depth - 1,
                reset_files=reset_files,
            )
            if not flagged:
                break

            category_subset = set().union(*flagged.values())
            row_texts: Dict[int, str] = {}
            for row_idx in flagged.keys():
                stripped = self._strip_snippets(original_texts[row_idx], aggregated.get(row_idx, {}))
                if stripped:
                    row_texts[row_idx] = stripped
            if not row_texts:
                break

            iteration_results = await self._gather_iteration(
                row_texts,
                original_texts=original_texts,
                raw_values=raw_values,
                categories=categories,
                additional_instructions=additional_instructions,
                iteration=depth,
                dynamic_mode=False,
                reset_files=False,
                category_subset=category_subset,
                **kwargs,
            )
            added = self._merge_snippet_results(aggregated, iteration_results)
            if not added:
                break

        return aggregated

    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        *,
        categories: Optional[Dict[str, str]] = None,
        additional_instructions: str = "",
        reset_files: bool = False,
        **kwargs: Any,
    ) -> pd.DataFrame:
        df_proc = df.reset_index(drop=True).copy()
        self.hit_rate_stats = {}

        raw_values = df_proc[column_name].tolist()
        original_texts = ["" if pd.isna(val) else str(val) for val in raw_values]

        additional = (additional_instructions or "").strip() or None
        dynamic_mode = categories is None
        if dynamic_mode and not additional:
            raise ValueError(
                "additional_instructions must be provided when categories is None"
            )

        categories_dict = categories or {}
        aggregated: Dict[int, Dict[str, List[str]]] = {}
        if not dynamic_mode:
            for idx in range(len(df_proc)):
                aggregated[idx] = {cat: [] for cat in categories_dict.keys()}

        row_texts = {idx: original_texts[idx] for idx in range(len(df_proc))}
        initial_results = await self._gather_iteration(
            row_texts,
            original_texts=original_texts,
            raw_values=raw_values,
            categories=None if dynamic_mode else categories_dict,
            additional_instructions=additional,
            iteration=0,
            dynamic_mode=dynamic_mode,
            reset_files=reset_files,
            **kwargs,
        )
        self._merge_snippet_results(aggregated, initial_results)

        if not dynamic_mode and categories_dict and self.cfg.n_rounds > 1:
            aggregated = await self._completion_loop(
                aggregated,
                original_texts,
                raw_values,
                categories_dict,
                additional,
                reset_files=reset_files,
                **kwargs,
            )

        if dynamic_mode:
            coded_passages: List[Dict[str, List[str]]] = []
            for idx in range(len(df_proc)):
                row_map = aggregated.get(idx, {})
                coded_passages.append(
                    {cat: list(snippets) for cat, snippets in row_map.items()}
                )
            df_proc["coded_passages"] = coded_passages
        else:
            for category in categories_dict.keys():
                df_proc[category] = [
                    list(aggregated.get(idx, {}).get(category, []))
                    for idx in range(len(df_proc))
                ]

        output_path = os.path.join(self.cfg.save_dir, "coded_passages.csv")
        df_proc.to_csv(output_path, index=False)

        if self.cfg.debug_print:
            print(f"\n[DEBUG] Processing complete. Results saved to: {self.cfg.save_dir}")
            self.print_final_hit_rates()

        return df_proc

