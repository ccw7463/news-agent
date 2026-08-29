"""FastMCP server exposing Google News search, headlines, and article reading.

Locale resolution is three-tiered, most specific first:

1. the ``language`` / ``region`` arguments on an individual tool call
2. the ``GOOGLE_RSS_LANGUAGE`` / ``GOOGLE_RSS_REGION`` environment variables
3. the neutral built-in defaults (``en`` / ``US``)

so an operator who only ever wants Korean articles sets the environment once,
while a shared deployment can still serve any locale per call.
"""

import logging
import os
from importlib.metadata import PackageNotFoundError, version
from typing import Annotated, Any, Dict, Literal, Optional

from fastmcp import FastMCP
from fastmcp.server.middleware.error_handling import ErrorHandlingMiddleware
from fastmcp.server.middleware.rate_limiting import RateLimitingMiddleware
from fastmcp.server.middleware.timing import TimingMiddleware
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import __version__ as _FALLBACK_VERSION
from .config import Settings
from .rss import TOPIC_PATHS, GoogleNewsClient

logger = logging.getLogger(__name__)

SETTINGS = Settings.from_env()

try:
    VERSION = version("google-rss-mcp")
except PackageNotFoundError:  # loaded from source, e.g. by a managed host
    VERSION = _FALLBACK_VERSION

Topic = Literal[
    "top",
    "world",
    "nation",
    "business",
    "technology",
    "entertainment",
    "sports",
    "science",
    "health",
]

# Guard against the enum above drifting from the client's routing table.
assert set(Topic.__args__) == set(TOPIC_PATHS), "Topic literal is out of sync"

mcp = FastMCP(
    name="google-rss-mcp",
    version=VERSION,
    instructions=(
        "Search Google News and read the resulting articles.\n\n"
        "Typical flow: call `search_news` or `get_top_headlines` to get "
        "headlines with real publisher URLs, then call `read_article` on the "
        "one or two that matter. Do not call `read_article` on every result — "
        "headlines alone answer most questions.\n\n"
        f"Articles default to language '{SETTINGS.language}' / region "
        f"'{SETTINGS.region}'. Pass `language` and `region` to override, e.g. "
        "language='ko', region='KR' for Korean coverage."
    ),
)

mcp.add_middleware(ErrorHandlingMiddleware())
mcp.add_middleware(RateLimitingMiddleware(max_requests_per_second=10))
mcp.add_middleware(TimingMiddleware())


def _client(language: Optional[str], region: Optional[str]) -> GoogleNewsClient:
    """Build a client for this call, applying the locale fallback chain.

    Args:
        language: Per-call language override, or ``None``.
        region: Per-call region override, or ``None``.

    Returns:
        A configured, unopened client.
    """
    return GoogleNewsClient(
        language=(language or SETTINGS.language).strip(),
        region=(region or SETTINGS.region).strip(),
        timeout=SETTINGS.timeout,
        max_concurrency=SETTINGS.max_concurrency,
    )


@mcp.tool(
    name="search_news",
    description=(
        "Search Google News for a keyword and return matching headlines with "
        "publisher URLs, sources, and timestamps. Returns headlines only — use "
        "read_article to get the body of a specific result."
    ),
)
async def search_news(
    query: Annotated[str, Field(description="Keyword or phrase to search for.")],
    max_results: Annotated[
        int, Field(ge=1, le=50, description="Number of headlines to return.")
    ] = 10,
    language: Annotated[
        Optional[str],
        Field(
            description="Language code, e.g. 'en', 'ko', 'ja'. Defaults to the server setting."
        ),
    ] = None,
    region: Annotated[
        Optional[str],
        Field(
            description="Region code, e.g. 'US', 'KR', 'JP'. Defaults to the server setting."
        ),
    ] = None,
    resolve_urls: Annotated[
        bool,
        Field(
            description="Resolve news.google.com links to publisher URLs. Slower but citable."
        ),
    ] = True,
) -> Dict[str, Any]:
    """Search Google News for a keyword.

    Args:
        query: Keyword or phrase to search for.
        max_results: Number of headlines to return.
        language: Per-call language override.
        region: Per-call region override.
        resolve_urls: Whether to resolve redirect links to publisher URLs.

    Returns:
        A dict with ``query``, ``language``, ``region``, ``count``, and
        ``articles`` (title, url, source, published).

    Raises:
        GoogleNewsError: If Google News is unreachable or returns bad data.
    """
    client = _client(language, region)
    async with client as news:
        items = await news.search(query, max_results, resolve_urls)
    return {
        "query": query,
        "language": client.language,
        "region": client.region,
        "count": len(items),
        "articles": [item.to_dict() for item in items],
    }


@mcp.tool(
    name="get_top_headlines",
    description=(
        "Get current headlines for a Google News topic section (top, world, "
        "business, technology, and so on). Returns headlines only — use "
        "read_article to get the body of a specific result."
    ),
)
async def get_top_headlines(
    topic: Annotated[Topic, Field(description="Topic section to fetch.")] = "top",
    max_results: Annotated[
        int, Field(ge=1, le=50, description="Number of headlines to return.")
    ] = 10,
    language: Annotated[
        Optional[str],
        Field(
            description="Language code, e.g. 'en', 'ko', 'ja'. Defaults to the server setting."
        ),
    ] = None,
    region: Annotated[
        Optional[str],
        Field(
            description="Region code, e.g. 'US', 'KR', 'JP'. Defaults to the server setting."
        ),
    ] = None,
    resolve_urls: Annotated[
        bool,
        Field(
            description="Resolve news.google.com links to publisher URLs. Slower but citable."
        ),
    ] = True,
) -> Dict[str, Any]:
    """Fetch headlines for a Google News topic section.

    Args:
        topic: Topic section to fetch.
        max_results: Number of headlines to return.
        language: Per-call language override.
        region: Per-call region override.
        resolve_urls: Whether to resolve redirect links to publisher URLs.

    Returns:
        A dict with ``topic``, ``language``, ``region``, ``count``, and
        ``articles`` (title, url, source, published).

    Raises:
        GoogleNewsError: If Google News is unreachable or returns bad data.
    """
    client = _client(language, region)
    async with client as news:
        items = await news.top_headlines(topic, max_results, resolve_urls)
    return {
        "topic": topic,
        "language": client.language,
        "region": client.region,
        "count": len(items),
        "articles": [item.to_dict() for item in items],
    }


@mcp.tool(
    name="read_article",
    description=(
        "Download one news article and return its readable text, title, and "
        "lead image. Accepts a publisher URL or a news.google.com link."
    ),
)
async def read_article(
    url: Annotated[str, Field(description="Article URL to read.")],
    max_length: Annotated[
        int,
        Field(
            ge=200, le=100_000, description="Maximum characters of body text to return."
        ),
    ] = SETTINGS.max_length,
) -> Dict[str, Any]:
    """Download one article and extract its text and lead image.

    Args:
        url: Article URL to read. May be a Google News redirect link.
        max_length: Maximum characters of body text to return.

    Returns:
        A dict with ``url``, ``title``, ``content``, ``image_url``, and
        ``truncated``.

    Raises:
        GoogleNewsError: If the page cannot be fetched.
    """
    async with _client(None, None) as news:
        article = await news.read_article(url, max_length)
    return article.to_dict()


@mcp.custom_route("/health", methods=["GET"])
async def health(_request: Request) -> JSONResponse:
    """Report liveness for platform health checks.

    Hosts (Koyeb, Cloud Run, Fly) probe an HTTP path and treat a non-2xx as a
    failed deploy. Every other path on this server is the MCP endpoint or a 404,
    so expose one cheap route that never touches the network.

    Args:
        _request: Unused; present to satisfy the Starlette route signature.

    Returns:
        200 with the server version and the locale defaults in force.
    """
    return JSONResponse(
        {
            "status": "ok",
            "server": "google-rss-mcp",
            "version": VERSION,
            "default_language": SETTINGS.language,
            "default_region": SETTINGS.region,
        }
    )


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean environment variable.

    Args:
        name: Variable name.
        default: Value to use when unset or unrecognized.

    Returns:
        The parsed flag.
    """
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def main() -> None:
    """Run the server over the transport named by ``MCP_TRANSPORT``.

    Defaults to ``stdio`` for local clients. Set ``MCP_TRANSPORT=http`` to serve
    Streamable HTTP on ``0.0.0.0:$PORT/mcp``, which is what a hosted deployment
    (and Smithery's URL publishing flow) needs.

    HTTP defaults to stateless mode. Session state lives in one process's
    memory, so on any autoscaled host — Cloud Run especially — a follow-up
    request routed to a second instance would fail to find its session. These
    three tools need no session state, so dropping it is free. Set
    ``MCP_STATELESS=false`` for a single-instance deployment that wants SSE
    resumability.
    """
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport in ("http", "streamable-http"):
        mcp.run(
            transport="http",
            host=os.environ.get("HOST", "0.0.0.0"),
            port=int(os.environ.get("PORT", "8081")),
            path="/mcp",
            stateless_http=_env_flag("MCP_STATELESS", True),
        )
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
