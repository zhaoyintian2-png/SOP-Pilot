# -*- coding: utf-8 -*-
"""
================================================================================
  config.py — Global Configuration
================================================================================
Centralizes all tunable constants for the SOP-Pilot project:
  - API / model settings (OpenRouter)
  - Ingestion parameters (URLs, chunk size, overlap, crawl depth)
  - Embedding backend selection
  - Safety / high-risk keyword list
  - LLM prompt templates
  - Cost accounting (API per-token pricing + human lookup cost baseline)
  - Dependency bootstrap (auto-install missing packages)

Every other module imports its constants from here so there is a single
source of truth.
================================================================================
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

# ----------------------------------------------------------------------------
# Dependency bootstrap
# ----------------------------------------------------------------------------
# If a local dependency directory exists (e.g. from --target install),
# add it to the search path first.
_LOCAL_DEPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".deps")
if os.path.isdir(_LOCAL_DEPS) and _LOCAL_DEPS not in sys.path:
    sys.path.insert(0, _LOCAL_DEPS)


def ensure_packages() -> None:
    """Ensure all required third-party packages are installed.

    Tries, in order: normal install -> --user -> local ``.deps`` directory.
    Called once at process start (see bottom of module).
    """
    required = [
        "openai",
        "scikit-learn",
        "sentence-transformers",
        "beautifulsoup4",
        "requests",
        "langchain",
        "langchain-community",
        "python-dotenv",
    ]
    import importlib

    missing: list[str] = []
    for pkg in required:
        import_name = {
            "scikit-learn": "sklearn",
            "beautifulsoup4": "bs4",
            "sentence-transformers": "sentence_transformers",
            "langchain-community": "langchain_community",
        }.get(pkg, pkg)
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return

    print(f"[Setup] Missing dependencies detected, attempting install: {missing}")
    last_err: Exception | None = None
    for extra in ([], ["--user"], ["--target", _LOCAL_DEPS]):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", *extra, *missing]
            )
            if "--target" in extra and _LOCAL_DEPS not in sys.path:
                sys.path.insert(0, _LOCAL_DEPS)
            print("[Setup] Dependencies installed successfully.")
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    print(
        f"[Setup] Auto-install failed: {last_err}\n"
        f"[Setup] Please run manually: pip install {' '.join(missing)}"
    )


# Run the bootstrap once so that downstream modules can import their deps.
ensure_packages()

# Load environment variables from a local .env file (if present).
# This lets users store their OPENROUTER_API_KEY in .env instead of exporting
# it manually every session. The .env file itself is gitignored.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # noqa: BLE001
    pass

# ============================================================================
# API / Model
# ============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default generation model (used by agent.generate when no override is given).
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"

# Default sampling temperature for answer generation.
DEFAULT_TEMPERATURE = 0.1


def get_openrouter_key() -> str:
    """Retrieve the OpenRouter API key.

    Priority: ``.env`` file (loaded into env) > environment variable
    ``OPENROUTER_API_KEY`` > interactive input.
    Raises ``RuntimeError`` if no key can be obtained.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        try:
            key = input("Enter your OpenRouter API Key: ").strip()
        except EOFError:
            key = ""
    if not key:
        raise RuntimeError(
            "No OpenRouter API Key provided. Please set the environment "
            "variable OPENROUTER_API_KEY before running."
        )
    return key


# ============================================================================
# Ingestion
# ============================================================================

# GitLab Handbook root URL (recursive crawl starting point).
ROOT_URL = "https://handbook.gitlab.com/handbook/"

# Recursive crawl parameters.
MAX_DEPTH = 3
MAX_PAGES = 300

# Date field bound to every chunk at ingest time.
SCRAPE_DATE = datetime.now().strftime("%Y-%m-%d")

# Fallback URL list used when RecursiveUrlLoader fails or returns empty.
FALLBACK_URLS = [
    "https://handbook.gitlab.com/handbook/people-group/",
    "https://handbook.gitlab.com/handbook/hiring/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/guide/",
    "https://handbook.gitlab.com/handbook/total-rewards/benefits/",
    "https://handbook.gitlab.com/handbook/paid-time-off/",
    "https://handbook.gitlab.com/handbook/communication/",
    "https://handbook.gitlab.com/handbook/company/culture/all-remote/asynchronous/",
    "https://handbook.gitlab.com/handbook/engineering/workflow/code-review/",
    "https://handbook.gitlab.com/handbook/engineering/workflow/",
    "https://handbook.gitlab.com/handbook/security/",
    "https://handbook.gitlab.com/handbook/security/security-assurance/",
    "https://handbook.gitlab.com/handbook/product/product-processes/",
]

# Browser User-Agent to avoid being blocked by the site's default UA filter.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
LOADER_HEADERS = {"User-Agent": BROWSER_UA}

# Chunking defaults.
CHUNK_SIZE = 100          # words per chunk (recursive-crawl path)
CHUNK_OVERLAP = 30        # sliding-window overlap in words
FALLBACK_CHUNK_SIZE = 150 # words per chunk (manual-URL fallback path)
MIN_CHUNK_TEXT_LEN = 50   # discard pages shorter than this many characters

# ============================================================================
# Embedding / Retrieval
# ============================================================================

# Preferred dense embedding model (free, offline, privacy-friendly).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Default number of top-k chunks to retrieve for a question.
DEFAULT_TOP_K = 3

# ============================================================================
# Safety (Human-in-the-Loop Interception)
# ============================================================================

# High-risk keywords that trigger automatic escalation to HR / Security.
HIGH_RISK_KEYWORDS = [
    "terminate", "termination", "fire ", "fired", "layoff", "lay off",
    "dismiss", "severance", "quit", "resign",
    "legal", "lawsuit", "sue", "compliance", "regulation", "gdpr",
    "liability", "contract breach", "nda",
    "harass", "harassment", "discrimination", "bullying", "hostile work",
    "credential", "leak", "security breach", "data breach", "phishing",
    "ransomware", "malware", "hack", "unauthorized access",
]

# Reply returned when a high-risk question is intercepted.
HIGH_RISK_REPLY = (
    "[HIGH-RISK SAFETY INTERCEPTION] Your question involves sensitive "
    "compliance or security concerns. It has been automatically escalated "
    "to HR / Security for manual handling. Please do not attempt to resolve "
    "this via self-service AI."
)

# Whitelist phrases: questions asking *how to report / policy for* are allowed
# through even if they contain risk-related words.
SAFETY_WHITELIST = ["how to report", "policy for", "what is the process"]

# ============================================================================
# Generation Prompts
# ============================================================================

SYSTEM_PROMPT = (
    "You are an Enterprise SOP assistant helping new employees. "
    "Answer using the notes provided below. "
    "If the notes contain relevant guidance, synthesize and answer based on them. "
    "If the notes genuinely do not contain any relevant information, reply exactly: "
    "The notes do not say. "
    "You MUST cite the source URL provided in the context at the end "
    "of your answer in the format: \n\nSource: <URL> "
    "(scraped on <date>). "
)

# ============================================================================
# Cost Accounting
# ============================================================================
# Per-token pricing (USD) for each supported model on OpenRouter.
# Prices are per single token (i.e. $/token), derived from the published
# $/1M-token rates. Update these to reflect current OpenRouter pricing.
MODEL_PRICING: dict[str, dict[str, float]] = {
    # deepseek flash-class models (approximate OpenRouter rates)
    "deepseek/deepseek-v4.1-flash": {
        "input": 0.14 / 1_000_000,
        "output": 0.28 / 1_000_000,
    },
    "deepseek/deepseek-chat": {
        "input": 0.07 / 1_000_000,
        "output": 0.28 / 1_000_000,
    },
    # openai models
    "openai/gpt-4o-mini": {
        "input": 0.15 / 1_000_000,
        "output": 0.60 / 1_000_000,
    },
    "openai/gpt-4o": {
        "input": 2.50 / 1_000_000,
        "output": 10.00 / 1_000_000,
    },
}

# Fallback price used when a model is not listed in MODEL_PRICING.
FALLBACK_PRICING = {"input": 1.00 / 1_000_000, "output": 3.00 / 1_000_000}

# --- Human lookup cost baseline ---------------------------------------------
# Cost of having a human (e.g. HR / department expert) manually look up an
# SOP answer. Used to compare against the AI API cost per query.
HUMAN_HOURLY_RATE_USD = 35.0           # loaded hourly cost of an HR/knowledge worker
HUMAN_LOOKUP_MINUTES = 8.0             # average minutes spent per manual lookup
HUMAN_LOOKUP_COST_USD = (
    HUMAN_HOURLY_RATE_USD * (HUMAN_LOOKUP_MINUTES / 60.0)
)

# ============================================================================
# Evaluation
# ============================================================================

EVAL_DATASET_PATH = "eval_dataset.json"
EVAL_OUTPUT_PATH = "eval_results.json"
EVAL_TOP_K = 10
