# -*- coding: utf-8 -*-
"""
================================================================================
  main.py — Application Entry Point
================================================================================
Orchestrates the full SOP-Pilot pipeline:

  1. Ingest  -> build the chunk corpus from the GitLab Handbook
                (ingestion.build_chunks).
  2. Index   -> build a Retriever over the corpus (retrieval.Retriever).
  3. Serve   -> construct the SOPAgent (agent.SOPAgent).
  4. Run     -> either run the evaluation suite (evaluate.evaluate) or an
                interactive Q&A loop.

Usage:
    python main.py                 # run the evaluation dataset
    python main.py --ask           # interactive Q&A mode
    python main.py --ask "question text"   # single question
================================================================================
"""

from __future__ import annotations

import argparse

from agent import SOPAgent
from evaluate import evaluate
from ingestion import build_chunks
from retrieval import Retriever


def build_agent() -> SOPAgent:
    """Ingest the corpus, build the retriever and return a ready-to-use agent."""
    chunks = build_chunks()
    print(f"\n[Ingest] Total chunks: {len(chunks)}")

    retriever = Retriever(chunks)
    agent = SOPAgent(retriever)
    return agent


def run_interactive(agent: SOPAgent, single_question: str | None = None) -> None:
    """Run an interactive Q&A loop (or answer a single question)."""
    if single_question:
        agent.answer(single_question)
        _print_summary(agent)
        return

    print("\n" + "=" * 70)
    print("  SOP-Pilot Interactive Mode  (type 'quit' or 'exit' to stop)")
    print("=" * 70)
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        agent.answer(question)
    _print_summary(agent)


def _print_summary(agent: SOPAgent) -> None:
    """Print accumulated token usage and cost totals."""
    usage = agent.get_token_usage()
    total_cost = agent.get_total_cost()
    human_cost = agent.human_lookup_cost()
    # number of answered (non-blocked, non-empty) queries approximated by total
    # cost / per-query average; instead report savings vs human at the same
    # query count.
    print("\n" + "#" * 70)
    print("[Session Summary]")
    print(f"  Prompt tokens     : {usage['prompt_tokens']}")
    print(f"  Completion tokens : {usage['completion_tokens']}")
    print(f"  Total tokens      : {usage['total_tokens']}")
    print(f"  Total API cost    : ${total_cost:.6f}")
    print(f"  Human baseline    : ${human_cost:.4f} / query")
    print("#" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="SOP-Pilot RAG Agent")
    parser.add_argument(
        "--ask",
        nargs="?",
        const="__interactive__",
        default=None,
        help="Ask a single question, or enter interactive mode if no question given.",
    )
    args = parser.parse_args()

    agent = build_agent()

    # Optional credit check before running (non-fatal).
    try:
        credit = agent.credit_check()
        print(f"[Credit Check] {credit}")
    except Exception as e:  # noqa: BLE001
        print(f"[Credit Check] Skipped ({e})")

    if args.ask is not None:
        single = None if args.ask == "__interactive__" else args.ask
        run_interactive(agent, single_question=single)
        return

    # Default: run the evaluation suite.
    evaluate(agent)


if __name__ == "__main__":
    main()
