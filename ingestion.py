# -*- coding: utf-8 -*-
"""
================================================================================
  ingestion.py — Web Scraping, Cleaning & Chunking
================================================================================
Responsibilities:
  - Fetch the GitLab Handbook (recursive crawl via LangChain RecursiveUrlLoader,
    with a manual URL-list fallback).
  - Extract clean main-body text (strip nav/footer/script noise).
  - Sliding-window chunk the text and bind each chunk to source URL + scrape
    date metadata.

Public entry point: :func:`build_chunks`.
================================================================================
"""

from __future__ import annotations

import itertools
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from config import (
    BROWSER_UA,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FALLBACK_CHUNK_SIZE,
    FALLBACK_URLS,
    LOADER_HEADERS,
    MAX_DEPTH,
    MAX_PAGES,
    MIN_CHUNK_TEXT_LEN,
    ROOT_URL,
    SCRAPE_DATE,
)


# ---------------------------------------------------------------------------
# HTML cleaning
# ---------------------------------------------------------------------------

def extract_main_content(html: str) -> str:
    """Extract clean main-body text from raw HTML.

    - Removes noise tags (script/style/nav/footer/header/aside/form/...).
    - Prioritizes ``<main>`` / ``<article>``, falls back to ``<body>``.
    - Collapses excess whitespace and blank lines.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        [
            "script", "style", "nav", "footer", "header", "aside",
            "noscript", "form", "button", "svg", "iframe",
        ]
    ):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.body
    if main is None:
        return ""

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Single-page fetch (used by the fallback path)
# ---------------------------------------------------------------------------

def fetch_text(url: str, timeout: int = 20) -> str:
    """Fetch a single page and return its cleaned body text."""
    try:
        resp = requests.get(
            url, headers={"User-Agent": BROWSER_UA}, timeout=timeout
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        print(f"[Fetch] Failed to fetch {url}: {e}")
        return ""
    return extract_main_content(resp.text)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window chunking by word count."""
    if not text:
        return []
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)]

    chunks: list[str] = []
    step = size - overlap
    if step <= 0:
        step = 1
    start = 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


# ---------------------------------------------------------------------------
# Document -> structured chunks
# ---------------------------------------------------------------------------

def _docs_to_chunks(docs: list[Any], scrape_date: str) -> list[dict]:
    """Convert a list of LangChain Documents into structured chunks.

    Each chunk dict: ``{"text", "url", "date", "doc_id"}``.
    Pages shorter than :data:`MIN_CHUNK_TEXT_LEN` are dropped (nav noise).
    """
    chunks: list[dict] = []
    for doc_id, doc in enumerate(docs):
        text = (getattr(doc, "page_content", "") or "").strip()
        url = (
            doc.metadata.get("source", "")
            if hasattr(doc, "metadata")
            else ""
        )
        if len(text) < MIN_CHUNK_TEXT_LEN:
            continue
        for piece in chunk(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
            chunks.append(
                {
                    "text": piece,
                    "url": url,
                    "date": scrape_date,
                    "doc_id": doc_id,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def build_chunks(
    root_url: str = ROOT_URL,
    max_depth: int = MAX_DEPTH,
) -> list[dict]:
    """Build the full chunk corpus.

    Strategy:
      1. Try LangChain ``RecursiveUrlLoader`` (recursive same-domain crawl).
      2. If it fails or returns nothing, fall back to fetching the manual
         :data:`FALLBACK_URLS` list page-by-page.
    """
    # ---- Option A: recursive crawl ----
    try:
        from langchain_community.document_loaders import RecursiveUrlLoader

        print(
            f"[Ingest] Using RecursiveUrlLoader (max_depth={max_depth}, "
            f"max_pages={MAX_PAGES}): {root_url}"
        )
        loader = RecursiveUrlLoader(
            url=root_url,
            max_depth=max_depth,
            extractor=extract_main_content,
            prevent_outside=True,
            use_async=False,
            timeout=20,
            headers=LOADER_HEADERS,
            check_response_status=True,
            continue_on_failure=True,
        )
        docs = list(itertools.islice(loader.lazy_load(), MAX_PAGES))

        print(f"[Ingest] Crawled {len(docs)} pages")
        chunks = _docs_to_chunks(docs, SCRAPE_DATE)
        if chunks:
            return chunks
        print("[Ingest] Recursive crawl returned empty, falling back to manual URL list.")
    except Exception as e:  # noqa: BLE001
        print(f"[Ingest] RecursiveUrlLoader failed, falling back to manual URL list: {e}")

    # ---- Option B: manual URL list ----
    print(f"[Ingest] Manually fetching {len(FALLBACK_URLS)} URLs ...")
    all_chunks: list[dict] = []
    for doc_id, url in enumerate(FALLBACK_URLS):
        print(f"[Ingest] Fetching ({doc_id + 1}/{len(FALLBACK_URLS)}): {url}")
        text = fetch_text(url)
        if not text:
            print(f"[Ingest] Skipping empty content: {url}")
            continue
        for piece in chunk(text, size=FALLBACK_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
            all_chunks.append(
                {
                    "text": piece,
                    "url": url,
                    "date": SCRAPE_DATE,
                    "doc_id": doc_id,
                }
            )
        n = sum(1 for c in all_chunks if c["doc_id"] == doc_id)
        print(f"[Ingest] Done, generated {n} chunks")
    return all_chunks
