import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app import ask_knowledgeguard, llm, users
from evaluation.evaluator import (
    evaluate_answer_completeness,
    evaluate_answer_relevance,
    evaluate_citation_accuracy,
    evaluate_context_sufficiency,
    evaluate_faithfulness,
    evaluate_hallucination_resistance,
    evaluate_permission_refusal,
)

console = Console()
CASES_PATH = Path(__file__).resolve().parent / "cases.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


def evaluate_single_case(case: dict, llm_judge: bool = True) -> dict:
    case_id = case["id"]
    user_id = case["user_id"]
    question = case["question"]
    expected_list = case.get("expected", [])
    forbidden_sources = case.get("forbidden_sources", [])
    is_deny_case = len(expected_list) == 0 and len(forbidden_sources) > 0
    is_unanswerable_case = len(expected_list) == 0 and len(forbidden_sources) == 0

    start_time = time.perf_counter()
    result = ask_knowledgeguard(question=question, user_id=user_id, return_details=True)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    answer = result.get("answer", "")
    context = result.get("context", [])

    retrieved_sources = [
        c.get("metadata", {}).get("source_path", "") for c in context
    ]

    # 1. Retrieval & ACL Check
    retrieval_pass = True
    retrieval_reasons = []

    # Check forbidden sources not present
    for forbidden in forbidden_sources:
        if forbidden in retrieved_sources:
            retrieval_pass = False
            retrieval_reasons.append(f"Forbidden source retrieved: {forbidden}")

    # Check expected content present for positive cases
    if not is_deny_case:
        all_content = " ".join(c.get("content", "").casefold() for c in context)
        for expected in expected_list:
            src = expected.get("source_path")
            if src and src not in retrieved_sources:
                retrieval_pass = False
                retrieval_reasons.append(f"Missing expected source: {src}")
            for phrase in expected.get("contains", []):
                if phrase.casefold() not in all_content:
                    retrieval_pass = False
                    retrieval_reasons.append(f"Missing expected phrase: {phrase}")

    # 2. Permission Refusal Check
    refusal_eval = evaluate_permission_refusal(context, answer, forbidden_sources)
    if is_deny_case and not refusal_eval["passed"]:
        retrieval_pass = False

    # 3. Citation Check
    citation_eval = evaluate_citation_accuracy(context, answer)

    # 4. Context Sufficiency Check
    all_expected_phrases = []
    for exp in expected_list:
        all_expected_phrases.extend(exp.get("contains", []))
    context_sufficiency_eval = evaluate_context_sufficiency(context, all_expected_phrases)

    # 5. Faithfulness & Relevance (LLM-as-a-Judge)
    faithfulness_eval = {"score": 1.0, "is_faithful": True, "reasoning": "Skipped LLM judge"}
    relevance_eval = {"score": 1.0, "is_relevant": True, "reasoning": "Skipped LLM judge"}
    completeness_eval = {"score": 1.0, "is_complete": True, "missing_parts": [], "reasoning": "Skipped LLM judge"}
    hallucination_eval = {"score": 1.0, "is_clean": True, "hallucinated_details": [], "reasoning": "Skipped LLM judge"}

    if llm_judge:
        if is_unanswerable_case and "i couldn't find relevant information in the available knowledge base." in answer.lower():
            faithfulness_eval = {
                "score": 1.0,
                "is_faithful": True,
                "unsupported_claims": [],
                "reasoning": "Correctly refused an unanswerable/out-of-scope query without fabricating facts.",
            }
        else:
            faithfulness_eval = evaluate_faithfulness(llm, question, context, answer)
        relevance_eval = evaluate_answer_relevance(
            llm,
            question,
            answer,
            is_refusal_expected=(is_deny_case or is_unanswerable_case),
        )
        completeness_eval = evaluate_answer_completeness(llm, question, answer)
        hallucination_eval = evaluate_hallucination_resistance(llm, context, answer)

    # Overall Verdict
    if is_deny_case:
        overall_pass = refusal_eval["passed"] and retrieval_pass
    elif is_unanswerable_case:
        overall_pass = (
            retrieval_pass
            and refusal_eval["refusal_message_present"]
            and hallucination_eval.get("is_clean", True)
        )
    else:
        overall_pass = (
            retrieval_pass
            and faithfulness_eval.get("is_faithful", False)
            and relevance_eval.get("is_relevant", True)
            and citation_eval["score"] >= 0.5
        )

    return {
        "case_id": case_id,
        "user_id": user_id,
        "is_deny_case": is_deny_case,
        "question": question,
        "answer": answer,
        "context_count": len(context),
        "latency_ms": round(latency_ms, 1),
        "retrieval_pass": retrieval_pass,
        "retrieval_reasons": retrieval_reasons,
        "refusal_eval": refusal_eval,
        "citation_eval": citation_eval,
        "context_sufficiency_eval": context_sufficiency_eval,
        "faithfulness_eval": faithfulness_eval,
        "relevance_eval": relevance_eval,
        "completeness_eval": completeness_eval,
        "hallucination_eval": hallucination_eval,
        "overall_pass": overall_pass,
    }


def run_benchmark(cases_to_run: list[dict], llm_judge: bool = True) -> list[dict]:
    results = []

    table = Table(
        title="KnowledgeGuard RAG & Permission Evaluation Matrix",
        border_style="cyan",
        header_style="bold magenta",
    )
    table.add_column("Case ID", style="bold", width=22)
    table.add_column("User", width=6)
    table.add_column("Type", width=8)
    table.add_column("Retrieval", width=11)
    table.add_column("Faithful", justify="center", width=10)
    table.add_column("Citations", justify="center", width=11)
    table.add_column("CtxSuff", justify="center", width=9)
    table.add_column("Complete", justify="center", width=10)
    table.add_column("NoHalluc", justify="center", width=10)
    table.add_column("Latency", justify="right", width=9)
    table.add_column("Verdict", justify="center", width=10)

    console.print(f"\n[cyan]Starting evaluation on {len(cases_to_run)} test cases...[/cyan]\n")

    for idx, case in enumerate(cases_to_run, 1):
        console.print(f"Running [{idx}/{len(cases_to_run)}]: {case['id']}...", end="\r")
        res = evaluate_single_case(case, llm_judge=llm_judge)
        results.append(res)

        case_type = "[yellow]DENY[/yellow]" if res["is_deny_case"] else "[blue]ANSWER[/blue]"
        ret_style = "[green]PASS[/green]" if res["retrieval_pass"] else "[red]FAIL[/red]"
        
        faith_score = res["faithfulness_eval"].get("score", 0.0)
        faith_style = f"[green]{faith_score:.2f}[/green]" if faith_score >= 0.8 else f"[red]{faith_score:.2f}[/red]"

        cit_score = res["citation_eval"].get("score", 0.0)
        cit_style = f"{cit_score:.2f}" if res["citation_eval"].get("has_sources_section") else "[dim]N/A[/dim]"

        ctx_suff_score = res["context_sufficiency_eval"].get("score", 0.0)
        ctx_suff_style = f"[green]{ctx_suff_score:.2f}[/green]" if ctx_suff_score >= 0.8 else f"[red]{ctx_suff_score:.2f}[/red]"

        compl_score = res["completeness_eval"].get("score", 0.0)
        compl_style = f"[green]{compl_score:.2f}[/green]" if compl_score >= 0.8 else f"[red]{compl_score:.2f}[/red]"

        halluc_score = res["hallucination_eval"].get("score", 0.0)
        halluc_style = f"[green]{halluc_score:.2f}[/green]" if halluc_score >= 0.9 else f"[red]{halluc_score:.2f}[/red]"

        lat_str = f"{res['latency_ms']:.0f} ms"
        verdict = "[bold green]PASS[/bold green]" if res["overall_pass"] else "[bold red]FAIL[/bold red]"

        table.add_row(
            res["case_id"],
            res["user_id"],
            case_type,
            ret_style,
            faith_style,
            cit_style,
            ctx_suff_style,
            compl_style,
            halluc_style,
            lat_str,
            verdict,
        )

    console.print()
    console.print(table)
    console.print()

    # Aggregate Metrics Scorecard
    total = len(results)
    passed = sum(1 for r in results if r["overall_pass"])
    acl_leak_count = sum(
        1 for r in results if len(r["refusal_eval"].get("leaked_forbidden_chunks", [])) > 0
    )
    deny_cases = [r for r in results if r["is_deny_case"]]
    deny_passed = sum(1 for r in deny_cases if r["overall_pass"]) if deny_cases else 0

    avg_faithfulness = (
        sum(r["faithfulness_eval"].get("score", 0.0) for r in results) / total if total else 0.0
    )
    avg_relevance = (
        sum(r["relevance_eval"].get("score", 0.0) for r in results) / total if total else 0.0
    )
    avg_context_sufficiency = (
        sum(r["context_sufficiency_eval"].get("score", 0.0) for r in results) / total if total else 0.0
    )
    avg_completeness = (
        sum(r["completeness_eval"].get("score", 0.0) for r in results) / total if total else 0.0
    )
    avg_hallucination_resistance = (
        sum(r["hallucination_eval"].get("score", 0.0) for r in results) / total if total else 0.0
    )
    avg_latency = (
        sum(r["latency_ms"] for r in results) / total if total else 0.0
    )

    scorecard = Table.grid(expand=True)
    scorecard.add_column(justify="center")
    scorecard.add_column(justify="center")
    scorecard.add_column(justify="center")
    scorecard.add_column(justify="center")

    acl_color = "green" if acl_leak_count == 0 else "red"
    acl_pct = 0.0 if acl_leak_count == 0 else (acl_leak_count / total) * 100
    c1 = Panel(f"[bold green]{passed}/{total}[/bold green]\nOverall Pass Rate", border_style="green")
    c2 = Panel(
        f"[bold {acl_color}]{acl_pct:.1f}%[/bold {acl_color}]\nACL Leakage Rate",
        border_style=acl_color,
    )
    c3 = Panel(
        f"[bold cyan]{avg_faithfulness*100:.1f}%[/bold cyan]\nAvg Faithfulness",
        border_style="cyan",
    )
    c4 = Panel(
        f"[bold magenta]{avg_latency:.0f} ms[/bold magenta]\nAvg Latency",
        border_style="magenta",
    )

    scorecard.add_row(c1, c2, c3, c4)

    scorecard2 = Table.grid(expand=True)
    scorecard2.add_column(justify="center")
    scorecard2.add_column(justify="center")
    scorecard2.add_column(justify="center")

    c5 = Panel(
        f"[bold yellow]{avg_context_sufficiency*100:.1f}%[/bold yellow]\nContext Sufficiency",
        border_style="yellow",
    )
    c6 = Panel(
        f"[bold blue]{avg_completeness*100:.1f}%[/bold blue]\nAnswer Completeness",
        border_style="blue",
    )
    c7 = Panel(
        f"[bold green]{avg_hallucination_resistance*100:.1f}%[/bold green]\nHallucination Resistance",
        border_style="green",
    )

    scorecard2.add_row(c5, c6, c7)

    console.print(scorecard)
    console.print(scorecard2)
    console.print()

    # Save to JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.time(),
                "total": total,
                "passed": passed,
                "acl_leak_count": acl_leak_count,
                "deny_cases_passed": f"{deny_passed}/{len(deny_cases)}",
                "avg_faithfulness": round(avg_faithfulness, 3),
                "avg_relevance": round(avg_relevance, 3),
                "avg_context_sufficiency": round(avg_context_sufficiency, 3),
                "avg_completeness": round(avg_completeness, 3),
                "avg_hallucination_resistance": round(avg_hallucination_resistance, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "results": results,
            },
            f,
            indent=2,
        )
    console.print(f"[green]Saved full evaluation results to {RESULTS_PATH}[/green]\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate KnowledgeGuard RAG & Permissions")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to run")
    parser.add_argument("--case", type=str, default=None, help="Run a specific test case by ID")
    parser.add_argument("--deny-only", action="store_true", help="Run only permission denial test cases")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip LLM judge calls to run faster")
    args = parser.parse_args()

    if not CASES_PATH.exists():
        console.print(f"[red]Cases file not found: {CASES_PATH}[/red]")
        sys.exit(1)

    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            console.print(f"[red]Case ID {args.case!r} not found.[/red]")
            sys.exit(1)

    if args.deny_only:
        cases = [c for c in cases if len(c.get("expected", [])) == 0 and len(c.get("forbidden_sources", [])) > 0]

    if args.limit:
        cases = cases[:args.limit]

    run_benchmark(cases, llm_judge=not args.no_llm_judge)


if __name__ == "__main__":
    main()
