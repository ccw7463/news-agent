"""Reading tool output back into article records.

The news server returns *headlines only* from ``search_news`` and
``get_top_headlines``; bodies come from a separate ``read_article`` call. That
split is what keeps a lookup to a few hundred tokens instead of tens of
thousands, and it is why this module exists: something has to decide which
headlines are worth spending a fetch on.
"""

import asyncio
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# langchain-mcp-adapters wraps a server-side tool error in its own prose.
# The part worth showing is what the server said.
_ADAPTER_PREFIX = re.compile(r"^(?:Internal error:\s*)?Error calling tool '[^']*':\s*")


def _coerce_payload(content: Any) -> Optional[Any]:
    """Turn one tool message's content into parsed JSON.

    The adapter hands back a plain string for a single text block, a list of
    content blocks otherwise, and occasionally an already-decoded object.

    Args:
        content: Raw ``ToolMessage.content``.

    Returns:
        The decoded payload, or ``None`` when it is not JSON we can use.
    """
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        # A list of content blocks carries the payload as JSON inside `text`.
        # It has to be unwrapped before the list itself is read as articles,
        # or every block turns into a titleless, urlless "article".
        blocks = [b for b in content if isinstance(b, dict) and b.get("text")]
        if len(blocks) == len(content) and blocks:
            for block in blocks:
                try:
                    return json.loads(block["text"])
                except json.JSONDecodeError:
                    continue
            return None
        return content

    return None


def _coerce_text(content: Any) -> str:
    """Flatten a tool result to plain text.

    A failed ``read_article`` does not raise — the adapter hands the server's
    message back as ordinary content. That message names the status code and
    says whether it is a paywall, which is the difference between "no news" and
    "this publisher blocks bots", so it must not be replaced with a generic one.

    Args:
        content: Raw tool result.

    Returns:
        The text, stripped of the adapter's wrapper prefix, or "".
    """
    if isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        text = str(content.get("text", ""))
    elif isinstance(content, list):
        text = " ".join(
            str(b.get("text", "")) for b in content if isinstance(b, dict)
        ).strip()
    else:
        text = ""

    return _ADAPTER_PREFIX.sub("", text).strip()


def extract_headlines(payload: Any) -> List[Dict[str, Any]]:
    """Pull article records out of one decoded tool payload.

    Accepts every shape the three tools can produce: the ``{count, articles}``
    envelope from a search or headline call, a bare ``read_article`` result, and
    a plain list, so a direct ``read_article`` call from the model is not lost.

    Args:
        payload: Decoded tool output.

    Returns:
        Article dicts, possibly empty.
    """
    if isinstance(payload, dict):
        articles = payload.get("articles")
        if isinstance(articles, list):
            return [a for a in articles if isinstance(a, dict)]
        # A read_article result: it carries a body, so it is already complete.
        if "url" in payload and "content" in payload:
            return [payload]
        return []

    if isinstance(payload, list):
        return [a for a in payload if isinstance(a, dict)]

    return []


def _strip_source_suffix(title: str, source: str) -> str:
    """Drop the trailing ``" - Publisher"`` Google News appends to every title.

    Fixed upstream in google-rss-mcp (and in the vendored copy), so on a current
    server this is a no-op. It stays because the agent can be pointed at any
    instance — an older deployment, or one that has not redeployed yet — and a
    surviving suffix costs tokens in every prompt and eats the visible half of
    the title in the sources table, which truncates. Idempotent either way.

    Args:
        title: Headline as received.
        source: Publisher name reported alongside it.

    Returns:
        The headline without the redundant publisher suffix.
    """
    if not source:
        return title
    suffix = f" - {source}"
    # Repeated, because the feed sometimes appends it twice — once by the
    # publisher in its own <title>, once by Google. Never down to nothing.
    stripped = title
    while stripped.endswith(suffix) and len(stripped) > len(suffix):
        candidate = stripped[: -len(suffix)].strip()
        if not candidate:
            break
        stripped = candidate
    return stripped


def collect_articles(tool_messages: Iterable[Any]) -> List[Dict[str, Any]]:
    """Gather deduplicated articles from every tool message in the run.

    Args:
        tool_messages: ``ToolMessage`` objects, in order.

    Returns:
        Article dicts, first occurrence wins, order preserved.
    """
    seen = set()
    collected = []

    for msg in tool_messages:
        payload = _coerce_payload(getattr(msg, "content", None))
        if payload is None:
            logger.warning("tool message was not JSON; skipping")
            continue
        for article in extract_headlines(payload):
            url = article.get("url") or article.get("article_url")
            key = url or article.get("title")
            if not key or key in seen:
                continue
            seen.add(key)
            record = dict(article)
            record["title"] = _strip_source_suffix(
                record.get("title") or "", record.get("source") or ""
            )
            collected.append(record)

    return collected


async def fetch_bodies(
    articles: List[Dict[str, Any]],
    read_article,
    limit: int,
    max_concurrency: int = 4,
    max_length: int = 5000,
    on_result=None,
) -> List[Dict[str, Any]]:
    """Fetch article bodies for the first ``limit`` articles that lack one.

    Only a slice is read on purpose. Fetching every headline would undo the
    reason the server splits search from reading, and publishers rate-limit.

    Args:
        articles: Records from :func:`collect_articles`, mutated in place.
        read_article: The bound ``read_article`` tool.
        limit: Maximum number of bodies to fetch.
        max_concurrency: Simultaneous fetches.
        max_length: Characters of body text to request per article.
        on_result: Optional ``callable(index, article, error)`` progress hook,
            where ``error`` is ``None`` on success.

    Returns:
        The articles that ended up with a usable body.
    """
    # A negative limit would slice [:-1] and read all but one — the opposite of
    # a budget.
    pending = [a for a in articles if not a.get("content")][: max(0, limit)]
    if not pending:
        return [a for a in articles if a.get("content")]

    semaphore = asyncio.Semaphore(max_concurrency)

    async def fetch(index: int, article: Dict[str, Any]) -> None:
        url = article.get("url") or article.get("article_url")
        if not url:
            article["read_error"] = "no url on this result"
            return
        async with semaphore:
            try:
                result = await read_article.ainvoke(
                    {"url": url, "max_length": max_length}
                )
            # One bad article must not sink the run. A transport-level failure
            # raises here; a publisher's 403 comes back as content instead, and
            # is handled below.
            except Exception as exc:  # noqa: BLE001
                article["read_error"] = f"{type(exc).__name__}: {exc}"
                if on_result:
                    on_result(index, article, article["read_error"])
                return

        # Not coalesced to {}: "well-formed but empty" and "not JSON at all" are
        # different failures with different messages.
        payload = _coerce_payload(result)
        if isinstance(payload, dict) and payload.get("content"):
            article["content"] = payload["content"]
            article["image_url"] = payload.get("image_url", "")
            article["truncated"] = payload.get("truncated", False)
            if payload.get("title") and not article.get("title"):
                article["title"] = payload["title"]
            if on_result:
                on_result(index, article, None)
        elif isinstance(payload, dict):
            # A well-formed result with an empty body: the fetch worked, the page
            # just had no extractable text. Reporting the raw JSON here would put
            # the whole document on the console behind a "✗".
            article["read_error"] = "the page had no extractable article text"
        else:
            # Not JSON at all, so it is the server's explanation — a 403 paywall,
            # a 404, a TLS failure. Keep it verbatim: "no news" and "this
            # publisher blocks bots" call for different next steps.
            article["read_error"] = (
                _coerce_text(result) or "no readable content returned"
            )
            if on_result:
                on_result(index, article, article["read_error"])

    await asyncio.gather(
        *(fetch(i, a) for i, a in enumerate(pending)), return_exceptions=True
    )
    return [a for a in articles if a.get("content")]
