import json
import re
from typing import Any, Optional

STANDARD_REFUSAL = "I couldn't find relevant information in the available knowledge base."


def evaluate_faithfulness(
    llm: Any,
    question: str,
    context_chunks: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """
    LLM-as-a-Judge: Assess whether every factual statement in the answer
    is strictly grounded in the retrieved context chunks.
    """
    if not context_chunks:
        is_refusal = STANDARD_REFUSAL.lower() in answer.lower()
        return {
            "score": 1.0 if is_refusal else 0.0,
            "is_faithful": is_refusal,
            "unsupported_claims": [] if is_refusal else ["Answered without retrieved context"],
            "reasoning": "Correct standard refusal on empty context." if is_refusal else "Answer generated despite empty context.",
        }

    context_text = "\n\n".join(
        f"[Source: {c.get('metadata', {}).get('source_path', 'unknown')}]\n{c.get('content', '')}"
        for c in context_chunks
    )

    judge_prompt = f"""You are a strict, objective AI evaluation judge.
Your task is to verify if the provided Answer is completely faithful to and grounded in the Retrieved Context.

Retrieved Context:
\"\"\"
{context_text}
\"\"\"

User Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Examine every factual claim in the Answer (excluding metadata in the Sources section).
2. Check if each claim is directly supported by the Retrieved Context.
3. If the Answer contains facts or claims not supported by the context, list them as unsupported.
4. Output your evaluation ONLY as a valid JSON object with the following schema:
{{
  "score": <float between 0.0 and 1.0>,
  "is_faithful": <true if score >= 0.8 else false>,
  "unsupported_claims": [<list of unsupported claim strings>],
  "reasoning": "<short explanation>"
}}
"""

    try:
        response = llm.invoke(judge_prompt)
        text = response.content.strip()
        # Extract JSON block if wrapped in markdown code fence
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "score": float(data.get("score", 0.0)),
                "is_faithful": bool(data.get("is_faithful", False)),
                "unsupported_claims": list(data.get("unsupported_claims", [])),
                "reasoning": str(data.get("reasoning", "")),
            }
    except Exception as e:
        return {
            "score": 0.5,
            "is_faithful": False,
            "unsupported_claims": [],
            "reasoning": f"Judge evaluation parsing error: {e}",
        }

    return {
        "score": 0.0,
        "is_faithful": False,
        "unsupported_claims": ["Failed to parse judge output"],
        "reasoning": "Judge output format invalid",
    }


def evaluate_answer_relevance(
    llm: Any,
    question: str,
    answer: str,
    is_refusal_expected: bool = False,
) -> dict[str, Any]:
    """
    LLM-as-a-Judge: Assess whether the answer directly addresses the user's question.
    """
    if is_refusal_expected:
        is_refusal = STANDARD_REFUSAL.lower() in answer.lower()
        return {
            "score": 1.0 if is_refusal else 0.0,
            "is_relevant": is_refusal,
            "reasoning": "Appropriately declined an unauthorized question." if is_refusal else "Failed to refuse an unauthorized query.",
        }

    judge_prompt = f"""You are a strict, objective AI evaluation judge.
Assess whether the Answer directly and helpfully answers the User Question.

User Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Rate relevance from 0.0 to 1.0:
   - 1.0: Fully answers the question clearly.
   - 0.5: Partially answers or is somewhat vague.
   - 0.0: Completely off-topic or fails to address the question.
2. Output your evaluation ONLY as a valid JSON object:
{{
  "score": <float between 0.0 and 1.0>,
  "is_relevant": <true if score >= 0.7 else false>,
  "reasoning": "<short explanation>"
}}
"""

    try:
        response = llm.invoke(judge_prompt)
        text = response.content.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "score": float(data.get("score", 0.0)),
                "is_relevant": bool(data.get("is_relevant", False)),
                "reasoning": str(data.get("reasoning", "")),
            }
    except Exception as e:
        return {
            "score": 0.5,
            "is_relevant": False,
            "reasoning": f"Relevance judge error: {e}",
        }

    return {
        "score": 0.0,
        "is_relevant": False,
        "reasoning": "Could not parse relevance output",
    }


def evaluate_citation_accuracy(
    context_chunks: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """
    Deterministic evaluation of the 'Sources:' section.
    Checks whether cited documents were actually present in the retrieved chunks.
    """
    if STANDARD_REFUSAL.lower() in answer.lower():
        # A refusal shouldn't have sources, which is valid
        return {
            "score": 1.0,
            "has_sources_section": False,
            "cited_sources": [],
            "valid_sources": [],
            "hallucinated_sources": [],
            "reasoning": "Refusal response correctly omitted citations.",
        }

    # Extract Sources section (handles "Sources:", "**Sources**", "### Sources", "**Sources:**", etc.)
    sources_match = re.search(
        r"(?:^|\n)\s*(?:\*{1,2}|#{1,4}\s*)?sources?(?:\*{1,2})?\s*:?\s*\n?(.*)",
        answer,
        re.IGNORECASE | re.DOTALL,
    )
    if not sources_match:
        return {
            "score": 0.0,
            "has_sources_section": False,
            "cited_sources": [],
            "valid_sources": [],
            "hallucinated_sources": [],
            "reasoning": "Missing mandatory 'Sources:' section in generated answer.",
        }

    sources_text = sources_match.group(1).strip()
    available_sources = {
        c.get("metadata", {}).get("source_path", "").casefold()
        for c in context_chunks
        if c.get("metadata", {}).get("source_path")
    }

    # Extract source lines or filenames
    raw_lines = [line.strip("- *• \t") for line in sources_text.splitlines() if line.strip("- *• \t")]
    if not raw_lines:
        return {
            "score": 0.0,
            "has_sources_section": True,
            "cited_sources": [],
            "valid_sources": [],
            "hallucinated_sources": [],
            "reasoning": "Empty Sources section.",
        }

    valid_sources = []
    hallucinated_sources = []

    for line in raw_lines:
        line_cf = line.casefold()
        # Check if any available source basename or path appears in this cited line
        matched = False
        for avail in available_sources:
            basename = avail.split("/")[-1]
            if basename in line_cf or avail in line_cf:
                matched = True
                valid_sources.append(line)
                break
        if not matched:
            hallucinated_sources.append(line)

    total_citations = len(valid_sources) + len(hallucinated_sources)
    score = (len(valid_sources) / total_citations) if total_citations > 0 else 0.0

    return {
        "score": round(score, 2),
        "has_sources_section": True,
        "cited_sources": raw_lines,
        "valid_sources": valid_sources,
        "hallucinated_sources": hallucinated_sources,
        "reasoning": f"{len(valid_sources)}/{total_citations} cited sources valid.",
    }


def evaluate_permission_refusal(
    retrieved_chunks: list[dict[str, Any]],
    answer: str,
    forbidden_sources: list[str],
) -> dict[str, Any]:
    """
    Verify security guarantees for an unauthorized query:
    1. Zero forbidden sources retrieved.
    2. Standard refusal message returned.
    """
    retrieved_sources = [
        c.get("metadata", {}).get("source_path", "")
        for c in retrieved_chunks
    ]

    leaked = [src for src in forbidden_sources if src in retrieved_sources]
    has_refusal = STANDARD_REFUSAL.lower() in answer.lower()

    passed = (len(leaked) == 0) and has_refusal

    return {
        "passed": passed,
        "leaked_forbidden_chunks": leaked,
        "refusal_message_present": has_refusal,
        "reasoning": "ACL enforced and refusal delivered cleanly." if passed else f"Security failure: leaked={leaked}, refusal={has_refusal}",
    }


def evaluate_context_sufficiency(
    context_chunks: list[dict[str, Any]],
    expected_phrases: list[str],
) -> dict[str, Any]:
    """
    Deterministic check that the retrieved context contains enough information
    to answer the question.  Goes beyond source-level matching: verifies that
    specific key phrases (from ``expected_phrases``) actually appear in the
    concatenated chunk text.

    Returns a dict with:
      - score (float 0-1): fraction of expected phrases found in context
      - found (list[str]): phrases that were found
      - missing (list[str]): phrases that were NOT found
      - reasoning (str): human-readable summary
    """
    if not expected_phrases:
        return {
            "score": 1.0,
            "found": [],
            "missing": [],
            "reasoning": "No expected phrases specified (deny/edge case).",
        }

    all_content = "\n".join(
        c.get("content", "") for c in context_chunks
    ).casefold()

    found: list[str] = []
    missing: list[str] = []

    for phrase in expected_phrases:
        if phrase.casefold() in all_content:
            found.append(phrase)
        else:
            missing.append(phrase)

    total = len(expected_phrases)
    score = len(found) / total if total > 0 else 0.0

    return {
        "score": round(score, 2),
        "found": found,
        "missing": missing,
        "reasoning": f"{len(found)}/{total} expected phrases found in context."
            + (f" Missing: {missing}" if missing else ""),
    }


def evaluate_answer_completeness(
    llm: Any,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """
    LLM-as-a-Judge: Assess whether the answer addresses ALL parts of a
    multi-part question.

    Particularly important for compound questions such as
    "What is X and what is Y?" — the answer must cover both X and Y.

    Returns a dict with:
      - score (float 0-1)
      - is_complete (bool): True when score >= 0.8
      - missing_parts (list[str]): sub-questions left unanswered
      - reasoning (str)
    """
    if not answer or not answer.strip():
        return {
            "score": 0.0,
            "is_complete": False,
            "missing_parts": ["Empty answer"],
            "reasoning": "No answer provided.",
        }

    if STANDARD_REFUSAL.lower() in answer.lower():
        return {
            "score": 1.0,
            "is_complete": True,
            "missing_parts": [],
            "reasoning": "Standard refusal — completeness N/A.",
        }

    judge_prompt = f"""You are a strict, objective AI evaluation judge.
Your task is to determine whether the Answer addresses ALL parts of the User Question.

User Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Break the User Question into its distinct sub-questions or informational requests.
2. For each sub-question, determine whether the Answer provides a clear response.
3. List any sub-questions that are not addressed or only partially addressed.
4. Output your evaluation ONLY as a valid JSON object with this schema:
{{
  "score": <float between 0.0 and 1.0 — fraction of sub-questions answered>,
  "is_complete": <true if score >= 0.8 else false>,
  "missing_parts": [<list of unanswered sub-question strings>],
  "reasoning": "<short explanation>"
}}
"""

    try:
        response = llm.invoke(judge_prompt)
        text = response.content.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "score": float(data.get("score", 0.0)),
                "is_complete": bool(data.get("is_complete", False)),
                "missing_parts": list(data.get("missing_parts", [])),
                "reasoning": str(data.get("reasoning", "")),
            }
    except Exception as e:
        return {
            "score": 0.5,
            "is_complete": False,
            "missing_parts": [],
            "reasoning": f"Completeness judge error: {e}",
        }

    return {
        "score": 0.0,
        "is_complete": False,
        "missing_parts": ["Failed to parse judge output"],
        "reasoning": "Judge output format invalid.",
    }


def evaluate_hallucination_resistance(
    llm: Any,
    context_chunks: list[dict[str, Any]],
    answer: str,
) -> dict[str, Any]:
    """
    LLM-as-a-Judge: Stricter than faithfulness — specifically tests whether
    the answer introduces plausible-sounding but unsupported details such as
    specific numbers, dates, names, or policy terms that do NOT appear in the
    retrieved context.

    This catches "confident hallucination" where the model invents realistic
    facts rather than obviously wrong ones.

    Returns a dict with:
      - score (float 0-1): 1.0 means zero hallucinated details
      - is_clean (bool): True when score >= 0.9
      - hallucinated_details (list[str]): invented facts found
      - reasoning (str)
    """
    if not answer or not answer.strip():
        return {
            "score": 1.0,
            "is_clean": True,
            "hallucinated_details": [],
            "reasoning": "No answer to evaluate.",
        }

    if STANDARD_REFUSAL.lower() in answer.lower():
        return {
            "score": 1.0,
            "is_clean": True,
            "hallucinated_details": [],
            "reasoning": "Standard refusal — no hallucination risk.",
        }

    if not context_chunks:
        has_substance = len(answer.strip()) > len(STANDARD_REFUSAL) + 20
        return {
            "score": 0.0 if has_substance else 1.0,
            "is_clean": not has_substance,
            "hallucinated_details": ["Answer produced without any context"] if has_substance else [],
            "reasoning": "Answer generated despite empty context." if has_substance else "Minimal answer with no context.",
        }

    context_text = "\n\n".join(
        f"[Source: {c.get('metadata', {}).get('source_path', 'unknown')}]\n{c.get('content', '')}"
        for c in context_chunks
    )

    judge_prompt = f"""You are a strict, meticulous AI evaluation judge specializing in hallucination detection.
Your task is to find ANY specific details in the Answer that are NOT present in or directly derivable from the Retrieved Context.

Retrieved Context:
\"\"\"
{context_text}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Instructions:
1. Examine every specific detail in the Answer: numbers, dates, names, percentages, policy terms, procedures, amounts, and conditions.
2. For each specific detail, verify it appears in or is directly derivable from the Retrieved Context.
3. Flag any detail that looks plausible but is NOT supported by the context — these are hallucinated details.
4. Ignore generic connective language, formatting, and the Sources section at the end.
5. Be strict: if a number or specific claim cannot be traced to the context, flag it.
6. Output your evaluation ONLY as a valid JSON object:
{{
  "score": <float 0.0 to 1.0; 1.0 = zero hallucinated details>,
  "is_clean": <true if score >= 0.9 else false>,
  "hallucinated_details": [<list of specific unsupported detail strings>],
  "reasoning": "<short explanation>"
}}
"""

    try:
        response = llm.invoke(judge_prompt)
        text = response.content.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            return {
                "score": float(data.get("score", 0.0)),
                "is_clean": bool(data.get("is_clean", False)),
                "hallucinated_details": list(data.get("hallucinated_details", [])),
                "reasoning": str(data.get("reasoning", "")),
            }
    except Exception as e:
        return {
            "score": 0.5,
            "is_clean": False,
            "hallucinated_details": [],
            "reasoning": f"Hallucination judge error: {e}",
        }

    return {
        "score": 0.0,
        "is_clean": False,
        "hallucinated_details": ["Failed to parse judge output"],
        "reasoning": "Judge output format invalid.",
    }
