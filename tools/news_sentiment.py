"""
tools/news_sentiment.py
Fetches recent headlines relevant to gold price drivers (Fed policy,
geopolitical risk, central bank buying). Prefers Tavily if an API key is
configured (higher quality, structured results); otherwise falls back to
DuckDuckGo's free search (`duckduckgo-search` / `ddgs` package).

This module never raises on a failed provider — it degrades to an empty
list and records a warning, since news is a "nice to have" enrichment and
should not block the rest of the pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

from config import APP_CONFIG, DATA_CONFIG
from schemas import NewsItem

logger = logging.getLogger(__name__)


def _fetch_via_tavily(query: str, max_results: int) -> list[NewsItem]:
    from tavily import TavilyClient  # local import: optional dependency

    client = TavilyClient(api_key=APP_CONFIG.tavily_api_key)
    resp = client.search(query=query, search_depth="basic", max_results=max_results, topic="news")
    items = []
    for r in resp.get("results", []):
        items.append(
            NewsItem(
                title=r.get("title", ""),
                source=r.get("url", "").split("/")[2] if r.get("url") else "",
                url=r.get("url", ""),
                published=r.get("published_date", ""),
                snippet=(r.get("content", "") or "")[:400],
            )
        )
    return items


def _fetch_via_duckduckgo(query: str, max_results: int) -> list[NewsItem]:
    try:
        from ddgs import DDGS  # newer package name
    except ImportError:
        from duckduckgo_search import DDGS  # legacy package name

    items = []
    with DDGS() as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            items.append(
                NewsItem(
                    title=r.get("title", ""),
                    source=r.get("source", ""),
                    url=r.get("url", ""),
                    published=r.get("date", ""),
                    snippet=(r.get("body", "") or "")[:400],
                )
            )
    return items


def fetch_news(max_results: Optional[int] = None, query_terms: Optional[tuple] = None) -> tuple[list[NewsItem], list[str]]:
    """Returns (news_items, warnings). Queries every configured term and
    de-duplicates by URL/title, capped at max_results overall."""
    max_results = max_results or DATA_CONFIG.news_max_results
    query_terms = query_terms or DATA_CONFIG.news_query_terms
    warnings: list[str] = []
    per_query = max(2, max_results // max(1, len(query_terms)))

    all_items: list[NewsItem] = []
    seen: set[str] = set()

    for term in query_terms:
        provider_items: list[NewsItem] = []
        try:
            if APP_CONFIG.tavily_api_key:
                provider_items = _fetch_via_tavily(term, per_query)
            else:
                provider_items = _fetch_via_duckduckgo(term, per_query)
        except Exception as e:
            warnings.append(f"News fetch failed for query '{term}': {e}")
            continue

        for item in provider_items:
            key = item.url or item.title
            if key and key not in seen:
                seen.add(key)
                all_items.append(item)

    if not all_items:
        warnings.append("No news items could be retrieved from any provider.")

    return all_items[:max_results], warnings
