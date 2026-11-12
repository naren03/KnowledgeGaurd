import math
import re
from collections import Counter
from typing import Any, Iterable

import numpy as np

_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "according",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "between",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "should",
        "that",
        "the",
        "their",
        "there",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "with",
    }
)


_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vacation": ("leave", "annual"),
    "headcount": ("employees",),
    "approval": ("approves", "approved", "approve"),
    "approves": ("approval", "approve"),
    "deployment": ("deployments", "release", "releases"),
    "deployments": ("deployment", "release", "releases"),
    "location": ("locations", "relocating", "relocate"),
}


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms excluding stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return [tok for tok in tokens if tok not in _STOPWORDS and (len(tok) > 1 or tok.isdigit())]


def _expand_query_tokens(tokens: list[str]) -> list[str]:
    """Add lightweight domain synonyms to query tokens for lexical recall."""
    expanded = list(tokens)
    seen = set(tokens)
    for tok in tokens:
        for syn in _QUERY_SYNONYMS.get(tok, ()):
            if syn not in seen:
                seen.add(syn)
                expanded.append(syn)
    return expanded


def _score_lexical_bm25(
    query_tokens: list[str],
    question_lower: str,
    candidates: list[tuple[int, dict[str, Any]]],
) -> list[float]:
    """Compute BM25 + filename + bigram lexical scores for authorized candidates."""
    if not query_tokens or not candidates:
        return [0.0] * len(candidates)

    n_docs = len(candidates)
    doc_term_freqs: list[Counter[str]] = []
    doc_lengths: list[int] = []
    df: Counter[str] = Counter()

    for _, chunk in candidates:
        content = str(chunk.get("content", ""))
        metadata = chunk.get("metadata", {})
        source_path = str(metadata.get("source_path", "")) if isinstance(metadata, dict) else ""

        content_tokens = _tokenize(content)
        source_tokens = _tokenize(source_path)

        tf: Counter[str] = Counter(content_tokens)
        for tok in source_tokens:
            tf[tok] += 2

        doc_term_freqs.append(tf)
        doc_len = len(content_tokens) + len(source_tokens)
        doc_lengths.append(doc_len)

        for tok in set(tf.keys()):
            df[tok] += 1

    avg_dl = (sum(doc_lengths) / n_docs) if n_docs > 0 else 1.0
    avg_dl = max(avg_dl, 1.0)
    k1 = 1.5
    b = 0.75

    # Build query bigrams for exact phrase bonus
    bigrams = [
        f"{query_tokens[i]} {query_tokens[i + 1]}"
        for i in range(len(query_tokens) - 1)
    ]

    unique_q = list(dict.fromkeys(_expand_query_tokens(query_tokens)))
    scores: list[float] = []

    for idx, (_, chunk) in enumerate(candidates):
        tf = doc_term_freqs[idx]
        dl = doc_lengths[idx]
        score = 0.0

        for tok in unique_q:
            freq = tf.get(tok, 0)
            if freq == 0:
                continue
            doc_freq = df.get(tok, 0)
            idf = math.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            denom = freq + k1 * (1.0 - b + b * (dl / avg_dl))
            score += idf * ((freq * (k1 + 1.0)) / denom)

        if bigrams and score > 0.0:
            content_norm = re.sub(r"\s+", " ", str(chunk.get("content", "")).casefold())
            for bg in bigrams:
                if bg in content_norm:
                    score += 0.6

        scores.append(score)

    return scores


def _rerank_hybrid(
    question: str,
    authorized_candidates: list[tuple[int, dict[str, Any]]],
    limit: int,
) -> list[dict[str, Any]]:
    """
    Combine FAISS dense rank with BM25 lexical rank via Reciprocal Rank Fusion
    and mild per-source diversity decay to support multi-document queries.
    """
    if not authorized_candidates:
        return []

    query_tokens = _tokenize(question)
    lex_scores = _score_lexical_bm25(query_tokens, question.casefold(), authorized_candidates)

    has_lexical_signal = any(s > 0.0 for s in lex_scores)
    if not has_lexical_signal:
        return [chunk for _, chunk in authorized_candidates[:limit]]

    # Rank indices with positive lexical score descending, tie-broken by dense_rank
    lex_ranked_indices = sorted(
        (i for i, s in enumerate(lex_scores) if s > 0.0),
        key=lambda i: (-lex_scores[i], authorized_candidates[i][0]),
    )
    lex_rank_map = {idx: rank for rank, idx in enumerate(lex_ranked_indices)}

    rrf_scores: list[float] = []
    for i, (dense_rank, _) in enumerate(authorized_candidates):
        score = 1.0 / (15.0 + dense_rank)
        if i in lex_rank_map:
            score += 1.6 / (15.0 + lex_rank_map[i])
        rrf_scores.append(score)

    # Greedy selection allowing up to 2 chunks per source before applying decay
    remaining = list(range(len(authorized_candidates)))
    selected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    while remaining and len(selected) < limit:
        best_pos = max(
            range(len(remaining)),
            key=lambda pos: rrf_scores[remaining[pos]]
            * (
                0.80
                ** max(
                    0,
                    source_counts[
                        str(
                            authorized_candidates[remaining[pos]][1]
                            .get("metadata", {})
                            .get("source_path", "unknown")
                        )
                    ]
                    - 2,
                )
            ),
        )
        chosen_idx = remaining.pop(best_pos)
        chosen_chunk = authorized_candidates[chosen_idx][1]
        src = str(chosen_chunk.get("metadata", {}).get("source_path", "unknown"))
        source_counts[src] += 1
        selected.append(chosen_chunk)

    return selected


def retrieve_chunks_with_audit(
    question: str,
    embedding_model: Any,
    index: Any,
    chunks: list[dict[str, Any]],
    user_groups: Iterable[str],
    limit: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Return highest-ranked chunks the user is allowed to read,
    along with detailed audit information regarding candidates scanned,
    authorized chunks, and denied chunks.
    """
    audit_data: dict[str, Any] = {
        "candidates_scanned": 0,
        "authorized_count": 0,
        "denied_count": 0,
        "authorized_sources": [],
        "denied_sources": [],
    }

    if limit <= 0 or getattr(index, "ntotal", 0) == 0 or not chunks:
        return [], audit_data

    question_vector = embedding_model.encode(
        [question], normalize_embeddings=True
    )
    question_vector = np.asarray(question_vector, dtype="float32")

    _, chunk_indexes = index.search(question_vector, k=index.ntotal)
    groups = set(user_groups)

    authorized_candidates: list[tuple[int, dict[str, Any]]] = []
    seen_denied_sources: set[str] = set()
    audit_window_open = True

    for chunk_index in chunk_indexes[0]:
        chunk_index = int(chunk_index)
        if chunk_index < 0 or chunk_index >= len(chunks):
            continue

        chunk = chunks[chunk_index]
        metadata = chunk.get("metadata", {})
        if not isinstance(metadata, dict):
            continue

        allowed_groups = metadata.get("allowed_groups", [])
        if not isinstance(allowed_groups, list):
            continue

        source_path = metadata.get("source_path", "unknown")
        is_authorized = any(
            isinstance(group, str) and group in groups for group in allowed_groups
        )

        if audit_window_open:
            audit_data["candidates_scanned"] += 1

        if is_authorized:
            authorized_candidates.append((len(authorized_candidates), chunk))
            if len(authorized_candidates) >= limit:
                audit_window_open = False
        elif audit_window_open:
            audit_data["denied_count"] += 1
            if source_path not in seen_denied_sources:
                seen_denied_sources.add(source_path)
                audit_data["denied_sources"].append(
                    {
                        "source_path": source_path,
                        "allowed_groups": allowed_groups,
                    }
                )

    selected_chunks = _rerank_hybrid(question, authorized_candidates, limit)
    audit_data["authorized_sources"] = [
        c.get("metadata", {}).get("source_path", "unknown") for c in selected_chunks
    ]
    audit_data["authorized_count"] = len(selected_chunks)
    return selected_chunks, audit_data


def retrieve_chunks(
    question: str,
    embedding_model: Any,
    index: Any,
    chunks: list[dict[str, Any]],
    user_groups: Iterable[str],
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Return the highest-ranked chunks the user's groups are allowed to read."""
    selected_chunks, _ = retrieve_chunks_with_audit(
        question=question,
        embedding_model=embedding_model,
        index=index,
        chunks=chunks,
        user_groups=user_groups,
        limit=limit,
    )
    return selected_chunks
