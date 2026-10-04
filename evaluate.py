# -*- coding: utf-8 -*-
"""
================================================================================
  evaluate.py — Evaluation Dataset & Metrics
================================================================================
Loads the evaluation dataset (``eval_dataset.json`` by default), runs the
:class:`SOPAgent` over every question, and computes pass/fail metrics plus an
aggregate cost-vs-human report.

A case is considered **PASS** when the generated answer:
  - does not refuse (no "the notes do not say" and no safety interception), AND
  - includes a source citation ("Source:").

Results (per-case log + aggregate metrics) are written to
``eval_results.json``.
================================================================================
"""

from __future__ import annotations

import json
from datetime import datetime

from agent import SOPAgent
from config import EVAL_DATASET_PATH, EVAL_OUTPUT_PATH, EVAL_TOP_K
from ingestion import build_chunks
from retrieval import Retriever


def load_dataset(path: str = EVAL_DATASET_PATH) -> list[dict]:
    """Load the evaluation dataset from ``path``.

    Falls back to a single default question if the file cannot be read.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[Eval] Loaded {len(data)} test questions from {path}")
        return data
    except Exception as e:  # noqa: BLE001
        print(f"[Eval] Failed to load {path}: {e}. Falling back to default question.")
        return [
            {
                "question": (
                    "What should a new employee do during onboarding at GitLab?"
                ),
                "answer_key": "Complete onboarding issue tasks.",
            }
        ]


def _is_passed(answer_text: str) -> bool:
    """Heuristic pass/fail check for a generated answer."""
    answer_lower = answer_text.lower()
    not_refused = (
        "the notes do not say" not in answer_lower
        and "interception" not in answer_lower
    )
    has_source = "source:" in answer_lower
    return not_refused and has_source


def evaluate(
    agent: SOPAgent,
    dataset_path: str = EVAL_DATASET_PATH,
    output_path: str = EVAL_OUTPUT_PATH,
    k: int = EVAL_TOP_K,
) -> dict:
    """Run the full evaluation and persist results.

    Returns a metrics dict with totals, pass rate and aggregate cost info.
    """
    eval_data = load_dataset(dataset_path)

    results_log: list[dict] = []
    passed_cases = 0
    total_api_cost = 0.0
    total_human_cost = 0.0

    print("\n[Evaluation] Starting evaluation on test dataset...")

    for idx, item in enumerate(eval_data, 1):
        q = item["question"]
        expected = item.get("answer_key", "")

        res = agent.answer(q, k=k)
        res["expected_answer_key"] = expected

        answer_text = res.get("answer", "")
        is_passed = _is_passed(answer_text)
        res["passed"] = is_passed

        if is_passed:
            passed_cases += 1
            status_str = "PASS"
        else:
            status_str = "FAIL"

        total_api_cost += res.get("api_cost_usd", 0.0)
        total_human_cost += agent.human_lookup_cost()

        results_log.append(res)
        print(f"[Progress] Case {idx}/{len(eval_data)} | Status: {status_str}")

    total_cases = len(results_log)
    pass_rate = (passed_cases / total_cases) * 100 if total_cases > 0 else 0
    total_savings = total_human_cost - total_api_cost
    total_savings_pct = (
        (total_savings / total_human_cost * 100) if total_human_cost else 0.0
    )

    output_payload = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "pass_rate_percentage": round(pass_rate, 2),
        "cost_summary": {
            "total_api_cost_usd": round(total_api_cost, 6),
            "total_human_cost_usd": round(total_human_cost, 4),
            "total_savings_usd": round(total_savings, 4),
            "total_savings_pct": round(total_savings_pct, 2),
        },
        "results": results_log,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, ensure_ascii=False, indent=2)

    print(f"\n" + "=" * 50)
    print("[Success] Evaluation complete!")
    print(f"  Total Cases  : {total_cases}")
    print(f"  Passed Cases : {passed_cases}")
    print(f"  Pass Rate    : {pass_rate:.2f}%")
    print(f"  Saved File   : {output_path}")
    print("-" * 50)
    print("[Cost Summary]")
    print(f"  Total API cost   : ${total_api_cost:.6f}")
    print(f"  Total human cost : ${total_human_cost:.4f}")
    print(f"  Total savings    : ${total_savings:.4f} ({total_savings_pct:.2f}%)")
    print("=" * 50)

    return output_payload


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
# Allows running ``python evaluate.py`` directly (without going through main.py).
# It builds the corpus + retriever + agent, then runs the evaluation.
if __name__ == "__main__":
    chunks = build_chunks()
    print(f"\n[Ingest] Total chunks: {len(chunks)}")

    retriever = Retriever(chunks)
    agent = SOPAgent(retriever)

    # Optional credit check (non-fatal).
    try:
        credit = agent.credit_check()
        print(f"[Credit Check] {credit}")
    except Exception as e:  # noqa: BLE001
        print(f"[Credit Check] Skipped ({e})")

    evaluate(agent)
