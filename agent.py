# -*- coding: utf-8 -*-
"""
================================================================================
  agent.py — LLM Generation + Cost Accounting
================================================================================
The :class:`SOPAgent` wires together retrieval, safety and LLM generation:

  1. :meth:`answer` runs the full RAG pipeline for a single question:
       safety scan -> retrieve top-k chunks -> build grounded prompt ->
       call LLM -> attach source citations.
  2. Every LLM call is metered. :meth:`calculate_api_cost` converts token
     consumption into USD using :data:`config.MODEL_PRICING`.
  3. :meth:`cost_comparison` contrasts the API cost of answering a query with
     the baseline cost of a human manually looking up the same SOP
     (:data:`config.HUMAN_LOOKUP_COST_USD`), so the ROI of the AI agent can
     be reported alongside every answer.

Token/cost totals are accumulated across the agent's lifetime and can be
inspected via :meth:`get_token_usage` and :meth:`get_total_cost`.
================================================================================
"""

from __future__ import annotations

from typing import Any

from config import (
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    FALLBACK_PRICING,
    HUMAN_LOOKUP_COST_USD,
    MODEL_PRICING,
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT,
    get_openrouter_key,
)
from retrieval import Retriever
from safety import check_high_risk


class SOPAgent:
    """Grounded SOP Q&A agent with per-call cost tracking."""

    def __init__(self, retriever: Retriever, model: str = DEFAULT_MODEL) -> None:
        self.retriever = retriever
        self.model = model

        # Lazy-initialized OpenRouter client (OpenAI-compatible).
        self._client: Any = None

        # Accumulated token usage across all generate() calls.
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        # Accumulated API cost in USD across all generate() calls.
        self._total_api_cost_usd = 0.0

    # ------------------------------------------------------------------
    # OpenRouter client
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=get_openrouter_key(),
                base_url=OPENROUTER_BASE_URL,
            )
        return self._client

    # ------------------------------------------------------------------
    # Cost accounting
    # ------------------------------------------------------------------

    def _pricing_for(self, model: str) -> dict[str, float]:
        return MODEL_PRICING.get(model, FALLBACK_PRICING)

    def calculate_api_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
    ) -> float:
        """Convert token counts into USD using the model's per-token pricing."""
        pricing = self._pricing_for(model or self.model)
        return (
            prompt_tokens * pricing["input"]
            + completion_tokens * pricing["output"]
        )

    @staticmethod
    def human_lookup_cost() -> float:
        """Baseline cost (USD) of a human manually answering one SOP query."""
        return HUMAN_LOOKUP_COST_USD

    def cost_comparison(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str | None = None,
    ) -> dict:
        """Build an API-vs-human cost report for a single LLM call.

        Returns a dict with:
          - ``api_cost_usd``
          - ``human_cost_usd``
          - ``savings_usd`` (human - api)
          - ``savings_pct`` (savings / human * 100)
        """
        api_cost = self.calculate_api_cost(
            prompt_tokens, completion_tokens, model=model
        )
        human_cost = self.human_lookup_cost()
        savings = human_cost - api_cost
        savings_pct = (savings / human_cost * 100) if human_cost else 0.0
        return {
            "api_cost_usd": api_cost,
            "human_cost_usd": human_cost,
            "savings_usd": savings,
            "savings_pct": round(savings_pct, 2),
        }

    def get_total_cost(self) -> float:
        return self._total_api_cost_usd

    def get_token_usage(self) -> dict:
        return dict(self.token_usage)

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> tuple[str, dict, float]:
        """Call OpenRouter and return ``(text, usage_dict, api_cost_usd)``.

        ``usage_dict`` contains ``prompt_tokens``, ``completion_tokens`` and
        ``total_tokens`` (zeros if the API did not report usage). Token counts
        and cost are also accumulated into the agent's running totals.
        """
        use_model = model or self.model
        client = self._get_client()
        response = client.chat.completions.create(
            model=use_model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", 0) or 0

        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] += total_tokens

        api_cost = self.calculate_api_cost(
            prompt_tokens, completion_tokens, model=use_model
        )
        self._total_api_cost_usd += api_cost

        usage_dict = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        text = response.choices[0].message.content or ""
        return text, usage_dict, api_cost

    def credit_check(self) -> dict:
        """Query the remaining quota for the current OpenRouter API key."""
        import requests

        key = get_openrouter_key()
        try:
            resp = requests.get(
                f"{OPENROUTER_BASE_URL}/auth/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            return {
                "limit": data.get("limit"),
                "usage": data.get("usage"),
                "remaining": data.get("remaining"),
                "rate_limit_requests": data.get("rate_limit_requests"),
            }
        except Exception as e:  # noqa: BLE001
            return {"error": f"Credit check failed: {e}"}

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context(results: list[dict]) -> tuple[str, list[dict]]:
        lines: list[str] = []
        sources: list[dict] = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Note {i}] (source: {r['url']}, date: {r['date']})\n{r['text']}"
            )
            sources.append(
                {"url": r["url"], "date": r["date"], "score": r["score"]}
            )
        return "\n\n".join(lines), sources

    # ------------------------------------------------------------------
    # Main answer pipeline
    # ------------------------------------------------------------------

    def answer(self, question: str, k: int = DEFAULT_TOP_K) -> dict:
        """Run the full safety -> retrieve -> generate pipeline for one question.

        Returns a result dict always containing ``question``, ``blocked`` and
        ``answer``, plus (on success) ``best_score``, ``sources``,
        ``token_usage``, ``api_cost_usd`` and ``cost_comparison``.
        """
        print("\n" + "=" * 70)
        print(f"[Question] {question}")
        print("=" * 70)

        # 1) Safety interception
        blocked = check_high_risk(question)
        if blocked is not None:
            print(f"\n{blocked}\n")
            return {
                "question": question,
                "blocked": True,
                "answer": blocked,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "api_cost_usd": 0.0,
                "cost_comparison": self.cost_comparison(0, 0),
            }

        # 2) Retrieve context
        results = self.retriever.retrieve(question, k=k)
        if not results:
            print("[Retrieve] No relevant chunks found.")
            return {
                "question": question,
                "blocked": False,
                "answer": "No context.",
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "api_cost_usd": 0.0,
                "cost_comparison": self.cost_comparison(0, 0),
            }

        best_score = results[0]["score"]
        context_text, sources = self._build_context(results)

        user_prompt = (
            f"Notes:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            "Answer using ONLY the notes above. Include the source URL at the end."
        )

        # 3) Generate
        try:
            response, usage, api_cost = self.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:  # noqa: BLE001
            response = f"[Generation Failed] Error calling the LLM: {e}"
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
            api_cost = 0.0
            print(f"\n[Answer]\n{response}\n")
            print(f"[Best Similarity Score] {best_score:.4f}")
            print("[Sources]")
            for s in sources:
                print(
                    f"  - {s['url']}  (scraped: {s['date']}, "
                    f"score: {s['score']:.4f})"
                )
            return {
                "question": question,
                "blocked": False,
                "answer": response,
                "best_score": best_score,
                "sources": sources,
                "token_usage": usage,
                "api_cost_usd": api_cost,
                "cost_comparison": self.cost_comparison(
                    usage["prompt_tokens"], usage["completion_tokens"]
                ),
                "error": str(e),
            }

        comparison = self.cost_comparison(
            usage["prompt_tokens"], usage["completion_tokens"]
        )

        print(f"\n[Answer]\n{response}\n")
        print(f"[Best Similarity Score] {best_score:.4f}")
        print("[Sources]")
        for s in sources:
            print(
                f"  - {s['url']}  (scraped: {s['date']}, "
                f"score: {s['score']:.4f})"
            )
        print(
            f"[Cost] API=${api_cost:.6f} | "
            f"Human=${comparison['human_cost_usd']:.4f} | "
            f"Savings=${comparison['savings_usd']:.4f} "
            f"({comparison['savings_pct']}%)"
        )

        return {
            "question": question,
            "blocked": False,
            "answer": response,
            "best_score": best_score,
            "sources": sources,
            "token_usage": usage,
            "api_cost_usd": api_cost,
            "cost_comparison": comparison,
        }
