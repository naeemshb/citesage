"""Evaluation Runner.

Runs the full evaluation pipeline:
1. Load test dataset
2. Run each question through the agent
3. Compute metrics
4. Generate evaluation report
5. Save results
"""

import json
import time
from datetime import datetime
from pathlib import Path

from src.agent.graph import run_agent
from src.evaluation.test_dataset import TEST_QUESTIONS
from src.evaluation.metrics import compute_all_metrics
from src.config import PROJECT_ROOT


def run_evaluation(
    questions: list[dict] | None = None,
    category_filter: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Run evaluation on test questions and compute aggregate metrics.

    Args:
        questions: Custom question list, or None to use TEST_QUESTIONS.
        category_filter: Only run questions of this category.
        output_path: Path to save results JSON.

    Returns:
        Evaluation report dict with per-question and aggregate metrics.
    """
    questions = questions or TEST_QUESTIONS

    if category_filter:
        questions = [q for q in questions if q["category"] == category_filter]

    print(f"Running evaluation on {len(questions)} questions...")
    print("=" * 60)

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q['question'][:80]}...")
        start = time.time()

        try:
            agent_result = run_agent(q["question"])
            elapsed = time.time() - start

            metrics = compute_all_metrics(
                generation=agent_result["answer"],
                ground_truth=q["ground_truth"],
                source_documents=agent_result["documents"],
                actual_route=agent_result["route"],
                expected_route=q["expected_route"],
                groundedness_score=agent_result["groundedness_score"],
            )

            result = {
                "question": q["question"],
                "category": q["category"],
                "expected_route": q["expected_route"],
                "actual_route": agent_result["route"],
                "answer": agent_result["answer"],
                "ground_truth": q["ground_truth"],
                "citations": agent_result["citations"],
                "trace": agent_result["trace"],
                "metrics": metrics,
                "latency_seconds": round(elapsed, 2),
                "status": "success",
            }

            status = "PASS" if metrics["groundedness_score"] >= 0.7 else "WARN"
            print(f"  [{status}] Route: {agent_result['route']}, "
                  f"Groundedness: {metrics['groundedness_score']:.2f}, "
                  f"Time: {elapsed:.1f}s")

        except Exception as e:
            result = {
                "question": q["question"],
                "category": q["category"],
                "status": "error",
                "error": str(e),
                "metrics": {},
                "latency_seconds": time.time() - start,
            }
            print(f"  [ERROR] {e}")

        results.append(result)

    # Aggregate metrics
    successful = [r for r in results if r["status"] == "success"]
    report = _build_report(results, successful)

    # Save
    if output_path is None:
        output_dir = PROJECT_ROOT / "evals" / "experiments"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"eval_{timestamp}.json"

    output_path = Path(output_path)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nResults saved to: {output_path}")

    _print_summary(report)
    return report


def _build_report(results: list[dict], successful: list[dict]) -> dict:
    """Build the evaluation report with aggregate metrics."""
    n = len(successful) if successful else 1

    # Per-metric averages
    metric_keys = [
        "citation_accuracy", "routing_accuracy", "answer_completeness",
        "groundedness_score",
    ]
    aggregates = {}
    for key in metric_keys:
        values = [r["metrics"][key] for r in successful if key in r["metrics"]]
        aggregates[key] = round(sum(values) / max(len(values), 1), 4)

    # Has citations rate
    has_cit = [r["metrics"].get("has_citations", False) for r in successful]
    aggregates["citation_rate"] = round(sum(has_cit) / n, 4)

    # Per-category breakdown
    categories = set(r["category"] for r in results)
    category_breakdown = {}
    for cat in categories:
        cat_results = [r for r in successful if r["category"] == cat]
        if cat_results:
            category_breakdown[cat] = {
                "count": len(cat_results),
                "avg_groundedness": round(
                    sum(r["metrics"]["groundedness_score"] for r in cat_results) / len(cat_results), 4
                ),
                "routing_accuracy": round(
                    sum(r["metrics"]["routing_accuracy"] for r in cat_results) / len(cat_results), 4
                ),
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(results),
        "successful": len(successful),
        "errors": len(results) - len(successful),
        "aggregate_metrics": aggregates,
        "category_breakdown": category_breakdown,
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in results) / max(len(results), 1), 2
        ),
        "results": results,
    }


def _print_summary(report: dict):
    """Print a human-readable summary of the evaluation."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total: {report['total_questions']} | "
          f"Success: {report['successful']} | "
          f"Errors: {report['errors']}")
    print(f"Avg latency: {report['avg_latency_seconds']}s")

    print("\nAggregate Metrics:")
    for key, val in report["aggregate_metrics"].items():
        print(f"  {key:.<30} {val:.4f}")

    print("\nPer-Category Breakdown:")
    for cat, data in report["category_breakdown"].items():
        print(f"  {cat}: n={data['count']}, "
              f"groundedness={data['avg_groundedness']:.2f}, "
              f"routing={data['routing_accuracy']:.2f}")
