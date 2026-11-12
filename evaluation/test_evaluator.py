import unittest

from evaluation.evaluator import (
    STANDARD_REFUSAL,
    evaluate_citation_accuracy,
    evaluate_context_sufficiency,
    evaluate_answer_completeness,
    evaluate_hallucination_resistance,
    evaluate_permission_refusal,
)


class EvaluatorTests(unittest.TestCase):
    def test_citation_accuracy_detects_valid_citations(self):
        context = [
            {
                "content": "Work from home is allowed 2 days a week.",
                "metadata": {"source_path": "docs/Remote_Work_Policy.pdf", "page": 1},
            }
        ]
        answer = (
            "Employees can work remotely up to 2 days per week.\n\n"
            "Sources:\n"
            "- Remote_Work_Policy.pdf, Page 1"
        )
        result = evaluate_citation_accuracy(context, answer)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(len(result["hallucinated_sources"]), 0)
        self.assertEqual(len(result["valid_sources"]), 1)

    def test_citation_accuracy_detects_hallucinated_citations(self):
        context = [
            {
                "content": "Lumetra headcount is 78.",
                "metadata": {"source_path": "docs/Company_Overview.pptx"},
            }
        ]
        answer = (
            "Lumetra has 78 employees.\n\n"
            "Sources:\n"
            "- Internal_Finance_Report.docx, Page 9"
        )
        result = evaluate_citation_accuracy(context, answer)
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(len(result["hallucinated_sources"]), 1)
        self.assertEqual(len(result["valid_sources"]), 0)

    def test_citation_accuracy_handles_missing_sources_section(self):
        context = [
            {
                "content": "Engineering uses Python and Go.",
                "metadata": {"source_path": "docs/Engineering_Handbook.pdf"},
            }
        ]
        answer = "Engineering uses Python and Go."
        result = evaluate_citation_accuracy(context, answer)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["has_sources_section"])

    def test_permission_refusal_passes_when_clean_and_refused(self):
        retrieved = []
        answer = STANDARD_REFUSAL
        forbidden = ["docs/Department_Budget.xlsx"]

        result = evaluate_permission_refusal(retrieved, answer, forbidden)
        self.assertTrue(result["passed"])
        self.assertEqual(result["leaked_forbidden_chunks"], [])
        self.assertTrue(result["refusal_message_present"])

    def test_permission_refusal_fails_when_forbidden_source_retrieved(self):
        retrieved = [
            {"metadata": {"source_path": "docs/Department_Budget.xlsx"}}
        ]
        answer = "The budget is $4.2M."
        forbidden = ["docs/Department_Budget.xlsx"]

        result = evaluate_permission_refusal(retrieved, answer, forbidden)
        self.assertFalse(result["passed"])
        self.assertEqual(result["leaked_forbidden_chunks"], ["docs/Department_Budget.xlsx"])


class FakeLLMResponse:
    """Minimal stub mimicking an LLM response object with a .content attribute."""
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """Fake LLM that returns a pre-configured JSON string."""
    def __init__(self, response_json: str):
        self._response_json = response_json

    def invoke(self, prompt: str) -> FakeLLMResponse:
        return FakeLLMResponse(self._response_json)


class FailingLLM:
    """LLM that always raises an exception."""
    def invoke(self, prompt: str):
        raise RuntimeError("LLM unavailable")


# ---------------------------------------------------------------------------
# Tests for evaluate_context_sufficiency
# ---------------------------------------------------------------------------
class ContextSufficiencyTests(unittest.TestCase):
    def test_all_phrases_found(self):
        chunks = [
            {"content": "Employees get 24 working days of annual leave and 12 paid sick days."}
        ]
        result = evaluate_context_sufficiency(chunks, ["24 working days of annual leave", "12 paid sick days"])
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(len(result["found"]), 2)
        self.assertEqual(len(result["missing"]), 0)

    def test_partial_phrases_found(self):
        chunks = [{"content": "Employees get 24 working days of annual leave."}]
        result = evaluate_context_sufficiency(chunks, ["24 working days of annual leave", "12 paid sick days"])
        self.assertEqual(result["score"], 0.5)
        self.assertEqual(result["found"], ["24 working days of annual leave"])
        self.assertEqual(result["missing"], ["12 paid sick days"])

    def test_no_phrases_found(self):
        chunks = [{"content": "This is about deployment procedures."}]
        result = evaluate_context_sufficiency(chunks, ["annual leave", "sick days"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(len(result["missing"]), 2)

    def test_empty_phrases_list(self):
        """Deny/edge cases pass with no expected phrases."""
        chunks = [{"content": "some content"}]
        result = evaluate_context_sufficiency(chunks, [])
        self.assertEqual(result["score"], 1.0)

    def test_empty_context_chunks(self):
        result = evaluate_context_sufficiency([], ["some phrase"])
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["missing"], ["some phrase"])

    def test_case_insensitive_matching(self):
        chunks = [{"content": "CI Must Be Green Before Merge"}]
        result = evaluate_context_sufficiency(chunks, ["CI must be green before merge"])
        self.assertEqual(result["score"], 1.0)

    def test_phrase_across_multiple_chunks(self):
        chunks = [
            {"content": "Employees get 24 working days of annual leave."},
            {"content": "They also receive 12 paid sick days."},
        ]
        result = evaluate_context_sufficiency(chunks, ["24 working days of annual leave", "12 paid sick days"])
        self.assertEqual(result["score"], 1.0)

    def test_missing_content_key(self):
        """Chunks with no 'content' key should not crash."""
        chunks = [{"metadata": {"source_path": "docs/test.txt"}}]
        result = evaluate_context_sufficiency(chunks, ["some phrase"])
        self.assertEqual(result["score"], 0.0)


# ---------------------------------------------------------------------------
# Tests for evaluate_answer_completeness
# ---------------------------------------------------------------------------
class AnswerCompletenessTests(unittest.TestCase):
    def test_complete_answer_via_llm(self):
        llm = FakeLLM('{"score": 1.0, "is_complete": true, "missing_parts": [], "reasoning": "All parts addressed."}')
        result = evaluate_answer_completeness(llm, "What is X and what is Y?", "X is foo and Y is bar.")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_complete"])
        self.assertEqual(result["missing_parts"], [])

    def test_incomplete_answer_via_llm(self):
        llm = FakeLLM('{"score": 0.5, "is_complete": false, "missing_parts": ["What is Y?"], "reasoning": "Only X answered."}')
        result = evaluate_answer_completeness(llm, "What is X and what is Y?", "X is foo.")
        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["is_complete"])
        self.assertIn("What is Y?", result["missing_parts"])

    def test_empty_answer(self):
        llm = FakeLLM('{}')  # should not be called
        result = evaluate_answer_completeness(llm, "Some question?", "")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["is_complete"])

    def test_whitespace_only_answer(self):
        llm = FakeLLM('{}')
        result = evaluate_answer_completeness(llm, "Some question?", "   \n  ")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["is_complete"])

    def test_refusal_answer(self):
        llm = FakeLLM('{}')
        result = evaluate_answer_completeness(llm, "Some question?", STANDARD_REFUSAL)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_complete"])

    def test_llm_error_returns_partial_score(self):
        llm = FailingLLM()
        result = evaluate_answer_completeness(llm, "What is X?", "X is something.")
        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["is_complete"])
        self.assertIn("error", result["reasoning"].lower())

    def test_malformed_llm_response(self):
        llm = FakeLLM("This is not JSON at all")
        result = evaluate_answer_completeness(llm, "What is X?", "X is something.")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["is_complete"])


# ---------------------------------------------------------------------------
# Tests for evaluate_hallucination_resistance
# ---------------------------------------------------------------------------
class HallucinationResistanceTests(unittest.TestCase):
    def test_clean_answer_via_llm(self):
        llm = FakeLLM('{"score": 1.0, "is_clean": true, "hallucinated_details": [], "reasoning": "All facts supported."}')
        context = [{"content": "Lumetra has 78 employees.", "metadata": {"source_path": "docs/Company_Overview.pptx"}}]
        result = evaluate_hallucination_resistance(llm, context, "Lumetra has 78 employees.")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_clean"])
        self.assertEqual(result["hallucinated_details"], [])

    def test_hallucinated_answer_via_llm(self):
        llm = FakeLLM('{"score": 0.3, "is_clean": false, "hallucinated_details": ["founded in 2019"], "reasoning": "Year not in context."}')
        context = [{"content": "Lumetra has 78 employees.", "metadata": {"source_path": "docs/Company_Overview.pptx"}}]
        result = evaluate_hallucination_resistance(llm, context, "Lumetra, founded in 2019, has 78 employees.")
        self.assertEqual(result["score"], 0.3)
        self.assertFalse(result["is_clean"])
        self.assertIn("founded in 2019", result["hallucinated_details"])

    def test_empty_answer(self):
        llm = FakeLLM('{}')
        context = [{"content": "some content", "metadata": {"source_path": "docs/test.txt"}}]
        result = evaluate_hallucination_resistance(llm, context, "")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_clean"])

    def test_refusal_answer(self):
        llm = FakeLLM('{}')
        context = []
        result = evaluate_hallucination_resistance(llm, context, STANDARD_REFUSAL)
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_clean"])

    def test_empty_context_with_substantive_answer(self):
        llm = FakeLLM('{}')
        long_answer = "Lumetra Technologies was founded in 2018 and has offices in three cities with over 200 employees."
        result = evaluate_hallucination_resistance(llm, [], long_answer)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["is_clean"])

    def test_empty_context_with_minimal_answer(self):
        llm = FakeLLM('{}')
        result = evaluate_hallucination_resistance(llm, [], "OK")
        self.assertEqual(result["score"], 1.0)
        self.assertTrue(result["is_clean"])

    def test_llm_error_returns_partial_score(self):
        context = [{"content": "Some content", "metadata": {"source_path": "docs/test.txt"}}]
        llm = FailingLLM()
        result = evaluate_hallucination_resistance(llm, context, "Some answer with details.")
        self.assertEqual(result["score"], 0.5)
        self.assertFalse(result["is_clean"])
        self.assertIn("error", result["reasoning"].lower())

    def test_malformed_llm_response(self):
        context = [{"content": "Some content", "metadata": {"source_path": "docs/test.txt"}}]
        llm = FakeLLM("not json")
        result = evaluate_hallucination_resistance(llm, context, "Some answer.")
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["is_clean"])


if __name__ == "__main__":
    unittest.main()

