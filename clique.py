import ast
import asyncio
import json
import os
import random
import re
import textwrap
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from utility_functions import Teleprompter, get_all_responses


def _swap_hierarchy_terms(
    hierarchy: Dict[str, Any], col1: str, col2: Optional[str]
) -> Dict[str, Any]:
    subs = [
        (re.compile(r"\bcircle passages\b", flags=re.I), f"{col1}"),
        (
            re.compile(r"\bsquare passages\b", flags=re.I),
            f"{col2}" if col2 else "square passages",
        ),
        (re.compile(r"\bpassage circle\b", flags=re.I), col1),
        (re.compile(r"\bpassage square\b", flags=re.I), col2 if col2 else "passage square"),
    ]

    def _swap_str(s: str) -> str:
        for pat, repl in subs:
            s = pat.sub(repl, s)
        return s

    def _walk(obj):
        if isinstance(obj, str):
            return _swap_str(obj)
        if isinstance(obj, dict):
            return {_walk(k): _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(x) for x in obj)
        if isinstance(obj, set):
            return {_walk(x) for x in obj}
        return obj

    return _walk(deepcopy(hierarchy))


class Clique:
    """
    Discover latent 'cliques' (or differentiators) with optional recursion.
    """

    # ───────────────── helper utils ─────────────────
    def __init__(self, teleprompter: Teleprompter) -> None:
        self.teleprompter = teleprompter

    _JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)

    @classmethod
    def _safe_json(cls, txt: Any) -> dict:
        """Parse LLM-emitted JSON (or Python-dict look-alike) into a real dict.

        Always returns a *dict*; on failure returns {}.
        Handles:
        • dict / list passthrough
        • markdown ```json fences
        • trailing commas
        • single-quoted keys / values
        • bare   selections: [...]  stanzas
        """
        # 0. identity / trivial cases -------------------------------------------------
        if isinstance(txt, dict):
            return txt
        if isinstance(txt, list):
            if txt and isinstance(txt[0], (dict, list, str)):
                return cls._safe_json(txt[0])
            return {}

        if not isinstance(txt, str):
            return {}

        s = txt.strip()

        # 1. unwrap ```json fences ----------------------------------------------------
        m = cls._JSON_FENCE_RE.search(s)
        if m:
            s = m.group(1).strip()

        # 2. wrap bare "selections:" payloads ----------------------------------------
        if re.match(r"^(\"?'?selections\"?'?\s*:)", s, re.I):
            s = "{" + s + "}"

        # 3. drop trailing commas -----------------------------------------------------
        s = re.sub(r",\s*([}\]])", r"\1", s)

        # 4. first try strict json ----------------------------------------------------
        try:
            return cls._safe_json(json.loads(s))
        except Exception:
            pass

        # 5. relax to Python literal eval --------------------------------------------
        try:
            return cls._safe_json(ast.literal_eval(s))
        except Exception:
            return {}

    @staticmethod
    def _unwrap(resp: Any) -> str:
        if isinstance(resp, list) and resp:
            return resp[0]
        if isinstance(resp, str):
            return resp
        return ""  # handles NaN / None

    @staticmethod
    def _grab_path(ident: str) -> str:
        parts = ident.split("|")
        if parts[1] == "buck":
            return parts[2]  # level|buck|PATH|…
        if parts[1] == "vote":
            return parts[3]  # level|vote|tag|PATH|…
        return parts[2] if len(parts) > 2 else ""

    async def _elo_step(
        self,
        texts: List[Tuple[str, str, Optional[str]]],
        bucket_terms: List[str],
        bucket_definitions: Dict[str, str],
        level: int,
        save_path: str,
        research_question: str,
        instructions: str,
        n_rounds: int,
        k_factor: float,
        n_parallels: int,
        model: str,
        reasoning_effort: str,
        use_dummy: bool,
        print_example_prompt: bool = False,
    ) -> pd.DataFrame:
        # initialize ratings
        ratings = {ident: {b: 1000.0 for b in bucket_terms} for ident, _, _ in texts}

        def expected(r_a, r_b):
            return 1 / (1 + 10 ** ((r_b - r_a) / 400))

        for rnd in range(n_rounds):
            round_path = os.path.join(save_path, f"level{level}_elo_round{rnd}.csv")
            if os.path.exists(round_path):
                resp_df = pd.read_csv(round_path)
            else:
                random.shuffle(texts)
                pairs = [(texts[i], texts[i + 1]) for i in range(0, len(texts) - 1, 2)]
                prompts, ids = [], []
                for (id_a, t_a, _), (id_b, t_b, _) in pairs:
                    circle_first = random.random() < 0.5
                    prompts.append(
                        self.teleprompter.clique_elo_prompt(
                            text_circle=t_a,
                            text_square=t_b,
                            bucket_terms=bucket_terms,
                            bucket_definitions=bucket_definitions,
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=None,
                            circle_first=circle_first,
                        )
                    )
                    ids.append(f"{rnd}|{id_a}|{id_b}")
                if not prompts:
                    continue
                resp_df = await get_all_responses(
                    prompts=prompts,
                    identifiers=ids,
                    n_parallels=n_parallels,
                    save_path=round_path,
                    reset_files=False,
                    json_mode=True,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    use_dummy=use_dummy,
                    print_example_prompt=print_example_prompt,
                )

            # ── update ratings
            for ident, resp in zip(resp_df.Identifier, resp_df.Response):
                try:
                    _, id_a, id_b = ident.split("|", 2)
                except ValueError:
                    continue
                # Robustly parse model output
                parsed = None
                if isinstance(resp, list) and resp:
                    parsed = self._safe_json(resp[0])
                elif isinstance(resp, str):
                    try:
                        import ast

                        resp_list = ast.literal_eval(resp)
                        if isinstance(resp_list, list) and resp_list:
                            parsed = self._safe_json(resp_list[0])
                    except Exception:
                        pass
                res = parsed or {}
                if id_a not in ratings or id_b not in ratings:
                    # Unexpected ids; skip this response entirely
                    continue
                for b, winner in res.items():
                    if b not in bucket_terms:
                        continue
                    if winner == "circle":
                        score_a, score_b = 1, 0
                    elif winner == "square":
                        score_a, score_b = 0, 1
                    else:
                        continue  # unrecognized winner value
                    exp_a = expected(ratings[id_a][b], ratings[id_b][b])
                    exp_b = 1 - exp_a
                    ratings[id_a][b] += k_factor * (score_a - exp_a)
                    ratings[id_b][b] += k_factor * (score_b - exp_b)

        rows = []
        for ident, text, label in texts:
            row = {"identifier": ident, "text": text}
            if label is not None:
                row["label"] = label
            for b in bucket_terms:
                row[b] = ratings[ident][b]
            rows.append(row)
        df_elo = pd.DataFrame(rows)
        df_elo.to_csv(os.path.join(save_path, f"level{level}_elo_final.csv"), index=False)
        return df_elo

    # ──────────────── main driver ────────────────
    async def run(
        self,
        df: pd.DataFrame,
        column_name: str,
        research_question: str,
        *,
        second_column_name: Optional[str] = None,
        recursion_depth: int = 1,
        instructions: str = "",
        n_cliques: int = 10,
        n_terms_per_round: int = 250,
        repeat_bucketing: int = 5,
        repeat_voting: int = 20,
        next_round_frac: float = 0.5,
        min_bucket_size: int = 1,
        n_parallels: int = 400,
        model: str = "o4-mini",
        reasoning_effort: str = "low",
        save_root: str = os.path.expanduser("~/Documents/runs"),
        run_name: Optional[str] = None,
        include_justifications: bool = True,
        reset_files: bool = False,
        debug_print: bool = False,
        use_dummy: bool = False,
        elo_rounds: int = 15,
        elo_k: float = 32.0,
        run_elo: bool = True,
        test_split: float = 0.0,
        print_example_prompt: bool = False,
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        # ── boilerplate
        if run_name is None:
            run_name = f"clique_run_{random.randint(0, 1_000_000)}"
        save_path = os.path.join(save_root, run_name)
        os.makedirs(save_path, exist_ok=True)

        df_proc = df.reset_index(drop=True).copy()
        if "extracted_terms" not in df_proc:
            df_proc["extracted_terms"] = [[] for _ in range(len(df_proc))]

        df_test = pd.DataFrame()
        if test_split > 0:
            df_test = df_proc.sample(frac=test_split, random_state=42)
            df_proc = df_proc.drop(df_test.index).reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

        hierarchy: Dict[str, Any] = {}
        metadata: Dict[str, Any] = {}

        Task = Tuple[int, Tuple[str, ...]]
        work_queue: List[Task] = [(i, tuple()) for i in range(len(df_proc))]
        test_queue: List[Task] = [(i, tuple()) for i in range(len(df_test))]

        # ── breadth‑first through recursion levels
        for level in range(1, recursion_depth + 1):
            if not work_queue:
                break

            ctx_to_rows: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
            for idx, path in work_queue:
                ctx_to_rows[path].append(idx)
            ctx_to_test_rows: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
            for idx, path in test_queue:
                ctx_to_test_rows[path].append(idx)

            # 1 ▸ extraction / differentiation
            prompts, ids = [], []
            for path, rows in ctx_to_rows.items():
                ctx_str = " » ".join(path) if path else None
                for i in rows:
                    if second_column_name is None:
                        p = self.teleprompter.clique_extraction_prompt(
                            text=str(df_proc.at[i, column_name]),
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=ctx_str,
                        )
                    else:
                        a = str(df_proc.at[i, column_name])
                        b = str(df_proc.at[i, second_column_name])
                        circle, square = a, b
                        circle_first = random.random() < 0.5
                        p = self.teleprompter.clique_differentiation_prompt(
                            text_circle=circle,
                            text_square=square,
                            instructions=instructions,
                            bucket_context=ctx_str,
                            circle_first=circle_first,
                        )
                    prompts.append(p)
                    ids.append(f"{i}|{level}|{'>>'.join(path)}")

            extract_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=n_parallels,
                save_path=os.path.join(save_path, f"level{level}_extraction.csv"),
                reset_files=reset_files,
                json_mode=True,
                model=model,
                use_dummy=use_dummy,
                print_example_prompt=print_example_prompt,
                reasoning_effort=reasoning_effort,
            )

            ctx_termjust: Dict[Tuple[str, ...], Dict[str, str]] = defaultdict(dict)
            for ident, resp in zip(extract_df.Identifier, extract_df.Response):
                row_idx = int(ident.split("|")[0])
                path = (
                    tuple(self._grab_path(ident).split(">>")) if self._grab_path(ident) else tuple()
                )
                parsed = self._safe_json(self._unwrap(resp))
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                if not isinstance(parsed, dict):
                    continue
                for t, j in parsed.items():
                    t_norm = t.lower().strip()
                    ctx_termjust[path][t_norm] = j  # case‑insensitive dedup
                df_proc.at[row_idx, "extracted_terms"] = list(parsed.keys())

            # 2 ▸ bucket proposal
            prompts, ids = [], []
            for path, term_just in ctx_termjust.items():
                terms = list(term_just.keys())
                for rep in range(repeat_bucketing):
                    random.shuffle(terms)
                    chunks = [
                        terms[i : i + n_terms_per_round]
                        for i in range(0, len(terms), n_terms_per_round)
                    ]
                    for ci, chunk in enumerate(chunks):
                        payload = [
                            {
                                "term": t,
                                "justification": term_just[t] if include_justifications else "",
                            }
                            for t in chunk
                        ]
                        prompts.append(
                            self.teleprompter.clique_bucket_prompt(
                                terms=payload,
                                n_cliques=n_cliques,
                                research_question=research_question,
                                instructions=instructions,
                                bucket_context=" » ".join(path) if path else None,
                                hierarchy=hierarchy,
                                second_column_name=second_column_name,
                            )
                        )
                        ids.append(f"{level}|buck|{'>>'.join(path)}|{rep}|{ci}")

            bucket_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=n_parallels,
                save_path=os.path.join(save_path, f"level{level}_bucket_gen.csv"),
                reset_files=reset_files,
                json_mode=True,
                model=model,
                max_retries=3,
                timeout=150,
                use_dummy=use_dummy,
                print_example_prompt=print_example_prompt,
                reasoning_effort=reasoning_effort,
            )

            ctx_candidates: Dict[Tuple[str, ...], set[str]] = defaultdict(set)
            ctx_bucketdef: Dict[Tuple[str, ...], Dict[str, str]] = defaultdict(dict)
            for ident, resp in zip(bucket_df.Identifier, bucket_df.Response):
                path = (
                    tuple(self._grab_path(ident).split(">>")) if self._grab_path(ident) else tuple()
                )
                parsed = self._safe_json(self._unwrap(resp))
                # If parsed is a list, take the first element
                if isinstance(parsed, list) and len(parsed) > 0:
                    parsed = parsed[0]
                if parsed and isinstance(parsed.get("buckets", {}), dict):
                    for b, j in parsed["buckets"].items():
                        ctx_candidates[path].add(b)
                        ctx_bucketdef[path][b] = j

            # helper to build vote prompts in one batch
            def _vote_prompts(ctx_opts, selected_map, tag, select_n):
                pr, idn = [], []
                for path, opts in ctx_opts.items():
                    if not opts:
                        continue
                    tjust = ctx_termjust[path]
                    bjust = ctx_bucketdef.get(path, {})
                    # Build selected bucket list for this path
                    sel_terms = selected_map.get(path, [])
                    sel_buckets = {b: bjust.get(b, "") for b in sel_terms}
                    for rep in range(repeat_voting):
                        random.shuffle(opts)
                        chunks = [
                            opts[i : i + n_terms_per_round]
                            for i in range(0, len(opts), n_terms_per_round)
                        ]
                        for ci, ch in enumerate(chunks):
                            sample = random.sample(
                                list(tjust.items()), min(n_terms_per_round, len(tjust))
                            )
                            payload_terms = [
                                {"term": t, "justification": j if include_justifications else ""}
                                for t, j in sample
                            ]
                            pr.append(
                                self.teleprompter.clique_vote_prompt(
                                    bucket_terms=ch,
                                    regular_terms=payload_terms,
                                    num_to_select=select_n,
                                    selected_buckets=sel_buckets,
                                    research_question=research_question,
                                    instructions=instructions,
                                    bucket_context=" » ".join(path) if path else None,
                                    hierarchy=hierarchy,
                                    bucket_definitions=bjust,
                                    second_column_name=second_column_name,
                                )
                            )
                            idn.append(f"{level}|vote|{tag}|{'>>'.join(path)}|{rep}|{ci}")
                return pr, idn

            # 3 ▸ iterative reduction
            ctx_current = {p: list(c) for p, c in ctx_candidates.items()}
            round_idx = 0
            while any(len(v) >= 3 * n_cliques for v in ctx_current.values()):
                round_idx += 1
                pr, idn = _vote_prompts(ctx_current, {}, f"reduce{round_idx}", n_cliques)
                vote_df = await get_all_responses(
                    prompts=pr,
                    identifiers=idn,
                    n_parallels=n_parallels,
                    save_path=os.path.join(save_path, f"level{level}_vote_reduce{round_idx}.csv"),
                    reset_files=reset_files,
                    json_mode=True,
                    model=model,
                    max_retries=3,
                    timeout=150,
                    use_dummy=use_dummy,
                    print_example_prompt=print_example_prompt,
                    reasoning_effort=reasoning_effort,
                )
                tallies: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
                    lambda: defaultdict(int)
                )
                for ident, resp in zip(vote_df.Identifier, vote_df.Response):
                    path = (
                        tuple(self._grab_path(ident).split(">>"))
                        if self._grab_path(ident)
                        else tuple()
                    )
                    parsed = self._safe_json(self._unwrap(resp))
                    if isinstance(parsed, list) and len(parsed) > 0:
                        parsed = parsed[0]
                    for b in parsed.get("voted_buckets", []) if parsed else []:
                        tallies[path][b] += 1
                for path, opts in ctx_current.items():
                    opts.sort(
                        key=lambda x: (tallies[path].get(x, 0), random.random()), reverse=True
                    )
                    ctx_current[path] = opts[: max(n_cliques, int(len(opts) * next_round_frac))]

            # 4 ▸ final pick loop
            ctx_final: Dict[Tuple[str, ...], List[str]] = {p: [] for p in ctx_current}
            for k in range(n_cliques):
                selected_map = {p: ctx_final[p] for p in ctx_current}
                pr, idn = _vote_prompts(
                    {p: [o for o in ctx_current[p] if o not in ctx_final[p]] for p in ctx_current},
                    selected_map=selected_map,
                    tag=f"final{k}",
                    select_n=n_cliques - k,
                )
                vote_df = await get_all_responses(
                    prompts=pr,
                    identifiers=idn,
                    n_parallels=n_parallels,
                    save_path=os.path.join(save_path, f"level{level}_vote_final{k}.csv"),
                    reset_files=reset_files,
                    json_mode=True,
                    model=model,
                    max_retries=3,
                    timeout=150,
                    use_dummy=use_dummy,
                    print_example_prompt=print_example_prompt,
                    reasoning_effort=reasoning_effort,
                )
                tallies: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(
                    lambda: defaultdict(int)
                )
                for ident, resp in zip(vote_df.Identifier, vote_df.Response):
                    path = (
                        tuple(self._grab_path(ident).split(">>"))
                        if self._grab_path(ident)
                        else tuple()
                    )
                    parsed = self._safe_json(self._unwrap(resp))
                    if isinstance(parsed, list) and len(parsed) > 0:
                        parsed = parsed[0]
                    for b in parsed.get("voted_buckets", []) if parsed else []:
                        tallies[path][b] += 1
                for path, counts in tallies.items():
                    remaining = [o for o in ctx_current[path] if o not in ctx_final[path]]
                    if remaining:
                        best = max(remaining, key=lambda x: (counts.get(x, 0), random.random()))
                        ctx_final[path].append(best)

            # Build canonical bucket_info for this level (term → definition)
            bucket_info_level: Dict[Tuple[str, ...], Dict[str, str]] = {
                path: {b: ctx_bucketdef[path][b] for b in buckets}
                for path, buckets in ctx_final.items()
            }

            # 5 ▸ apply unified prompt
            prompts, ids = [], []
            for path, rows in ctx_to_rows.items():
                order = random.sample(list(bucket_info_level[path]), len(bucket_info_level[path]))
                defs_map = bucket_info_level[path]
                ctx_str = " » ".join(path) if path else None
                for i in rows:
                    if second_column_name is None:
                        p = self.teleprompter.clique_apply_prompt(
                            text=str(df_proc.at[i, column_name]),
                            bucket_terms=order,
                            bucket_definitions=defs_map,
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=ctx_str,
                        )
                    else:
                        a = str(df_proc.at[i, column_name])
                        b = str(df_proc.at[i, second_column_name])
                        circle, square = a, b
                        circle_first = random.random() < 0.5
                        p = self.teleprompter.clique_apply_prompt(
                            text=circle,
                            text_square=square,
                            bucket_terms=order,
                            bucket_definitions=defs_map,
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=ctx_str,
                            circle_first=circle_first,
                        )
                    prompts.append(p)
                    ids.append(f"{i}|{level}|{'>>'.join(path)}")

            apply_df = await get_all_responses(
                prompts=prompts,
                identifiers=ids,
                n_parallels=n_parallels,
                save_path=os.path.join(save_path, f"level{level}_apply.csv"),
                reset_files=reset_files,
                json_mode=True,
                model=model,
                use_dummy=use_dummy,
                print_example_prompt=print_example_prompt,
                reasoning_effort=reasoning_effort,
            )

            # test set classification
            test_prompts, test_ids = [], []
            for path, rows in ctx_to_test_rows.items():
                if not ctx_final.get(path):
                    continue
                order = random.sample(ctx_final[path], len(ctx_final[path]))
                defs_map = {b: ctx_bucketdef[path][b] for b in order}
                ctx_str = " » ".join(path) if path else None
                for i in rows:
                    if second_column_name is None:
                        p = self.teleprompter.clique_apply_prompt(
                            text=str(df_test.at[i, column_name]),
                            bucket_terms=order,
                            bucket_definitions=defs_map,
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=ctx_str,
                        )
                    else:
                        a = str(df_test.at[i, column_name])
                        b = str(df_test.at[i, second_column_name])
                        circle_first = random.random() < 0.5
                        p = self.teleprompter.clique_apply_prompt(
                            text=a,
                            text_square=b,
                            bucket_terms=order,
                            bucket_definitions=defs_map,
                            research_question=research_question,
                            instructions=instructions,
                            bucket_context=ctx_str,
                            circle_first=circle_first,
                        )
                    test_prompts.append(p)
                    test_ids.append(f"{i}|{level}|{'>>'.join(path)}")

            if test_prompts:
                test_apply_df = await get_all_responses(
                    prompts=test_prompts,
                    identifiers=test_ids,
                    n_parallels=n_parallels,
                    save_path=os.path.join(save_path, f"level{level}_apply_test.csv"),
                    reset_files=reset_files,
                    json_mode=True,
                    model=model,
                    use_dummy=use_dummy,
                    print_example_prompt=print_example_prompt,
                    reasoning_effort=reasoning_effort,
                )

            # — write selections & queue children
            next_queue: List[Task] = []
            col_name = f"level_{level}_buckets"
            if col_name not in df_proc:
                df_proc[col_name] = [[] for _ in range(len(df_proc))]

            def _norm(s: str) -> str:
                """
                Canonicalise a bucket label so that punctuation & dash / slash / quote
                differences never block a match.
                """
                import re
                import unicodedata

                # 1️⃣  Unicode-fold & lowercase
                s = unicodedata.normalize("NFKD", s).lower()

                # 2️⃣  Convert every dash variant & every slash into a space.
                #      Hyphen/minus is placed at the *end* of the [] so it isn’t parsed as a range.
                s = re.sub(r"[‐-‒–—−/\\-]", " ", s)

                # 3️⃣  Strip all quotes / apostrophes
                s = re.sub(r"[\"'‘’“”`]", "", s)

                # 4️⃣  Collapse whitespace and drop any remaining punctuation
                s = re.sub(r"\s+", " ", s).strip()
                s = re.sub(r"[^a-z0-9 ]", "", s)

                return s

            for ident, resp in zip(apply_df.Identifier, apply_df.Response):
                row_idx = int(ident.split("|")[0])
                path = (
                    tuple(self._grab_path(ident).split(">>")) if self._grab_path(ident) else tuple()
                )
                parsed = self._safe_json(self._unwrap(resp))
                chosen_raw = parsed.get("selections", []) if parsed else []
                canon_map = {_norm(b): b for b in bucket_info_level[path]}
                chosen = []
                for c in chosen_raw:
                    key = _norm(c)
                    if key == "none of these":
                        continue
                    if key in canon_map:
                        chosen.append(canon_map[key])
                df_proc.at[row_idx, col_name] = chosen

                if level < recursion_depth:
                    for c in chosen:
                        next_queue.append((row_idx, (*path, c)))

            if debug_print:
                unused = {b for b in bucket_list if not any(b in x for x in df_proc[col_name])}
                if unused:
                    print("Buckets never selected:", sorted(list(unused))[:10], "…")

            # handle test set results
            next_test_queue: List[Task] = []
            if test_prompts:
                if col_name not in df_test:
                    df_test[col_name] = [[] for _ in range(len(df_test))]
                for ident, resp in zip(test_apply_df.Identifier, test_apply_df.Response):
                    row_idx = int(ident.split("|")[0])
                    path = (
                        tuple(self._grab_path(ident).split(">>"))
                        if self._grab_path(ident)
                        else tuple()
                    )
                    parsed = self._safe_json(self._unwrap(resp))
                    chosen_raw = parsed.get("selections", []) if parsed else []
                    canon_map = {_norm(b): b for b in bucket_info_level[path]}
                    chosen = []
                    for c in chosen_raw:
                        key = _norm(c)
                        if key == "none of these":
                            continue
                        if key in canon_map:
                            chosen.append(canon_map[key])
                    df_test.at[row_idx, col_name] = chosen

                    if level < recursion_depth:
                        for c in chosen:
                            next_test_queue.append((row_idx, (*path, c)))
            else:
                next_test_queue = test_queue

            # — extend hierarchy using bucket_info_level
            for path, buckets_map in bucket_info_level.items():
                node = hierarchy
                for part in path:
                    node = node.setdefault(part, {})
                node["_buckets"] = buckets_map

            # — map each bucket term to a directional attribute (one prompt per term)
            dir_prompts, dir_ids = [], []
            for path, buckets_map in bucket_info_level.items():
                for term, definition in buckets_map.items():
                    dir_prompts.append(
                        self.teleprompter.clique_directional_prompt(
                            bucket_term=term,
                            bucket_definition=definition,
                        )
                    )
                    # Encode term in identifier to recover later
                    dir_ids.append(f"{level}|dir|{'>>'.join(path)}|{term}")
            # path → {bucket_term → directional_attr}
            bucket_dirs: Dict[Tuple[str, ...], Dict[str, str]] = {}
            # path → {directional_attr → definition}
            bucket_dir_defs: Dict[Tuple[str, ...], Dict[str, str]] = {}
            if dir_prompts:
                dir_df = await get_all_responses(
                    prompts=dir_prompts,
                    identifiers=dir_ids,
                    n_parallels=n_parallels,
                    save_path=os.path.join(save_path, f"level{level}_directions.csv"),
                    reset_files=reset_files,
                    json_mode=True,
                    model=model,
                    use_dummy=use_dummy,
                    print_example_prompt=print_example_prompt,
                    reasoning_effort=reasoning_effort,
                )
                for ident, resp in zip(dir_df.Identifier, dir_df.Response):
                    path = (
                        tuple(self._grab_path(ident).split(">>"))
                        if self._grab_path(ident)
                        else tuple()
                    )
                    # bucket term is the last field in the identifier
                    term = ident.split("|")[-1]

                    parsed = self._safe_json(self._unwrap(resp))
                    if isinstance(parsed, list) and len(parsed) > 0:
                        parsed = parsed[0]
                    if isinstance(parsed, dict) and len(parsed) == 1:
                        # Expecting exactly one directional attribute
                        direction_attr, definition = next(iter(parsed.items()))
                        # Build mappings incrementally
                        bucket_dirs.setdefault(path, {})[term] = direction_attr
                        bucket_dir_defs.setdefault(path, {})[direction_attr] = definition
                    else:
                        # Fallback: if structure different, attempt to parse attribute/definition keys
                        if isinstance(parsed, dict):
                            attr = parsed.get("attribute")
                            defin = parsed.get("definition")
                            if attr:
                                bucket_dirs.setdefault(path, {})[term] = attr
                                if defin:
                                    bucket_dir_defs.setdefault(path, {})[attr] = defin
                metadata.setdefault("directional_mappings", {})[f"level_{level}"] = bucket_dirs
                metadata.setdefault("directional_definitions", {})[f"level_{level}"] = (
                    bucket_dir_defs
                )

            read_col = f"{col_name}_directional"
            if read_col not in df_proc:
                df_proc[read_col] = [[] for _ in range(len(df_proc))]
            if not df_test.empty and read_col not in df_test:
                df_test[read_col] = [[] for _ in range(len(df_test))]
            for path, rows in ctx_to_rows.items():
                for i in rows:
                    chosen_terms = df_proc.at[i, col_name]
                    converted = []
                    for term in chosen_terms:
                        t = bucket_dirs.get(path, {}).get(term, term)
                        t = re.sub(re.compile("circle", re.IGNORECASE), column_name, t)
                        if second_column_name is not None:
                            t = re.sub(re.compile("square", re.IGNORECASE), second_column_name, t)
                        converted.append(t)
                    df_proc.at[i, read_col] = converted
            for path, rows in ctx_to_test_rows.items():
                for i in rows:
                    chosen_terms = df_test.at[i, col_name]
                    converted = []
                    for term in chosen_terms:
                        t = bucket_dirs.get(path, {}).get(term, term)
                        t = re.sub(re.compile("circle", re.IGNORECASE), column_name, t)
                        if second_column_name is not None:
                            t = re.sub(re.compile("square", re.IGNORECASE), second_column_name, t)
                        converted.append(t)
                    df_test.at[i, read_col] = converted

            if run_elo:
                texts: List[Tuple[str, str, Optional[str]]] = []
                for rows in ctx_to_rows.values():
                    for i in rows:
                        if second_column_name is None:
                            texts.append((str(i), str(df_proc.at[i, column_name]), column_name))
                        else:
                            texts.append((f"{i}_a", str(df_proc.at[i, column_name]), column_name))
                            texts.append(
                                (
                                    f"{i}_b",
                                    str(df_proc.at[i, second_column_name]),
                                    second_column_name,
                                )
                            )

                # ── build bucket_list: keep *only* directional attrs, dedup by canon key ──
                def _canon(s: str) -> str:
                    """Cheap normaliser to merge punctuation variants."""
                    import re
                    import unicodedata

                    s = unicodedata.normalize("NFKD", s).lower()
                    s = re.sub(r"[‐-‒–—−/\\]", " ", s)  # dashes & slashes → space
                    s = re.sub(r"[\"'‘’“”`]", "", s)  # drop quotes / apostrophes
                    s = re.sub(r"\s+", " ", s).strip()
                    return s

                canon_to_attr: dict[str, str] = {}  # later paths overwrite earlier

                for path, bl in ctx_final.items():
                    dir_map = bucket_dirs.get(path, {})  # {bucket_term → directional_attr}
                    for b in bl:
                        attr = dir_map.get(b)  # None if no directional mapping
                        if attr is None:
                            continue  # ✱ skip non-directional buckets ✱
                        canon_to_attr[_canon(attr)] = attr  # last spelling wins

                bucket_list = sorted(canon_to_attr.values())
                # ───────────────────────────────────────────────────────────────────────────

                bucket_defs_map = {}
                for path, bl in ctx_final.items():
                    for b in bl:
                        attr = bucket_dirs.get(path, {}).get(b, b)
                        defin = bucket_dir_defs.get(path, {}).get(attr) or ctx_bucketdef[path][b]
                        bucket_defs_map[attr] = defin
                if bucket_list and texts:
                    await self._elo_step(
                        texts=texts,
                        bucket_terms=bucket_list,
                        bucket_definitions=bucket_defs_map,
                        level=level,
                        save_path=save_path,
                        research_question=research_question,
                        instructions=instructions,
                        n_rounds=elo_rounds,
                        k_factor=elo_k,
                        n_parallels=n_parallels,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        use_dummy=use_dummy,
                        print_example_prompt=print_example_prompt,
                    )
                    metadata.setdefault("elo_paths", {})[f"level_{level}"] = os.path.join(
                        save_path, f"level{level}_elo_final.csv"
                    )

                if bucket_list and not df_test.empty:
                    test_texts: List[Tuple[str, str, Optional[str]]] = []
                    for rows in ctx_to_test_rows.values():
                        for i in rows:
                            if second_column_name is None:
                                test_texts.append(
                                    (f"test{i}", str(df_test.at[i, column_name]), column_name)
                                )
                            else:
                                test_texts.append(
                                    (f"test{i}_a", str(df_test.at[i, column_name]), column_name)
                                )
                                test_texts.append(
                                    (
                                        f"test{i}_b",
                                        str(df_test.at[i, second_column_name]),
                                        second_column_name,
                                    )
                                )
                    if test_texts:
                        await self._elo_step(
                            texts=test_texts,
                            bucket_terms=bucket_list,
                            bucket_definitions=bucket_defs_map,
                            level=level,
                            save_path=os.path.join(save_path, "test"),
                            research_question=research_question,
                            instructions=instructions,
                            n_rounds=elo_rounds,
                            k_factor=elo_k,
                            n_parallels=n_parallels,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            use_dummy=use_dummy,
                            print_example_prompt=print_example_prompt,
                        )

            # — prune undersized
            work_queue = [
                t
                for t in next_queue
                if sum(1 for x in next_queue if x[1] == t[1]) >= min_bucket_size
            ]
            test_queue = [
                t
                for t in next_test_queue
                if sum(1 for x in next_test_queue if x[1] == t[1]) >= min_bucket_size
            ]

        # ── create readable bucket columns (apply swaps) ──────────
        def _make_readable_label(lbl):
            """Convert circle/square placeholders to column names.

            Works on strings **or** on (nested) lists/tuples of strings, returning the same
            container type with converted entries. All other types are returned untouched.
            """
            if isinstance(lbl, str):
                subs_local = [
                    (re.compile(r"\bcircle passages\b", flags=re.I), f"{column_name}"),
                    (
                        re.compile(r"\bsquare passages\b", flags=re.I),
                        f"{second_column_name}" if second_column_name else "square passages",
                    ),
                    (re.compile(r"\bpassage circle\b", flags=re.I), column_name),
                    (
                        re.compile(r"\bpassage square\b", flags=re.I),
                        second_column_name if second_column_name else "passage square",
                    ),
                ]
                for pat, repl in subs_local:
                    lbl = pat.sub(repl, lbl)
                return lbl
            elif isinstance(lbl, list):
                return [_make_readable_label(x) for x in lbl]
            elif isinstance(lbl, tuple):
                return tuple(_make_readable_label(x) for x in lbl)
            else:
                return lbl

        level_cols = [c for c in df_proc.columns if c.startswith("level_")]
        for col in level_cols:
            df_proc[f"{col}_readable"] = df_proc[col].apply(_make_readable_label)
            if not df_test.empty:
                df_test[f"{col}_readable"] = df_test[col].apply(_make_readable_label)

        df_proc.to_csv(os.path.join(save_path, "final_classifications.csv"), index=False)
        import json

        with open(os.path.join(save_path, "hierarchy_raw.json"), "w") as f:
            json.dump(hierarchy, f, indent=2)
        readable_hierarchy = _swap_hierarchy_terms(hierarchy, column_name, second_column_name)
        with open(os.path.join(save_path, "hierarchy_readable.json"), "w") as f:
            json.dump(readable_hierarchy, f, indent=2)
        if not df_test.empty:
            df_test.to_csv(os.path.join(save_path, "final_classifications_test.csv"), index=False)
        return hierarchy, df_proc, df_test, metadata
