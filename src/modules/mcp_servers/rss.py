"""Async client for Google News RSS feeds and article extraction.

The client is deliberately dependency-light: ``aiohttp`` for transport,
``feedparser`` for the feed, ``beautifulsoup4`` + ``html2text`` for readable
article text. Locale is a constructor argument, so the same process can serve
callers in different languages.
"""

import asyncio
import html
import json
import logging
import random
import re
import ssl
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin, urlparse

import aiohttp
import certifi
import feedparser
import html2text
from bs4 import BeautifulSoup

try:  # feedparser keeps its date parser in a private module that has moved before.
    from feedparser.datetimes import _parse_date as _feedparser_parse_date
except ImportError:  # pragma: no cover - depends on feedparser internals
    _feedparser_parse_date = None

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

GOOGLE_NEWS_HOST = "news.google.com"
BATCH_EXECUTE_URL = "https://news.google.com/_/DotsSplashUi/data/batchexecute"

# Google News topic sections. Keep in sync with ``server.Topic``.
TOPIC_PATHS: Dict[str, Optional[str]] = {
    "top": None,  # the feed root is the top-stories feed
    "world": "WORLD",
    "nation": "NATION",
    "business": "BUSINESS",
    "technology": "TECHNOLOGY",
    "entertainment": "ENTERTAINMENT",
    "sports": "SPORTS",
    "science": "SCIENCE",
    "health": "HEALTH",
}

# Page furniture that must not end up in the extracted article body.
_BOILERPLATE_TAGS = (
    "script",
    "style",
    "noscript",
    "iframe",
    "form",
    "nav",
    "header",
    "footer",
    "aside",
    "svg",
    "button",
)

# Ordered by how likely the container is to hold the real article body.
_ARTICLE_SELECTORS = (
    "article",
    '[itemprop="articleBody"]',
    ".article-body",
    ".story-body",
    ".entry-content",
    ".post-content",
    "#article-body",
    '[role="main"]',
    "main",
)

_IMAGE_SKIP_PATTERNS = (
    "sprite",
    "placeholder",
    "avatar",
    "logo",
    "icon",
    "banner",
    "1x1",
    "pixel",
    "spacer",
    "blank",
)


class GoogleNewsError(RuntimeError):
    """Raised when Google News cannot be reached or returns unusable data."""


def _strip_source_suffix(title: str, source: str) -> str:
    """Remove the trailing ``" - Publisher"`` Google appends to every headline.

    Google sends the publisher twice: once in the ``<source>`` element and again
    glued to the end of ``<title>``. Repeating the name costs tokens in every
    prompt it reaches, and clients that truncate a headline for display lose the
    part that carries the meaning.

    Stripped repeatedly, because the suffix can appear twice — a publisher whose
    own ``<title>`` already ends with its name gets Google's copy on top. Never
    strips the title down to nothing.

    Args:
        title: Headline as Google sent it.
        source: Publisher name from the feed's ``<source>`` element.

    Returns:
        The headline without the redundant publisher suffix.
    """
    if not source:
        return title
    suffix = f" - {source}"
    while title.endswith(suffix):
        candidate = title[: -len(suffix)].strip()
        if not candidate:
            break
        title = candidate
    return title


@dataclass
class NewsItem:
    """A single headline from an RSS feed.

    Attributes:
        title: Headline text, with the trailing " - Source" suffix removed.
        url: Article URL. Resolved to the publisher when ``resolve_urls`` is on,
            otherwise the ``news.google.com`` redirect link.
        source: Publisher name as reported by Google News.
        published: Publication timestamp, or ``None`` when unparseable.
    """

    title: str
    url: str
    source: str = ""
    published: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe primitives."""
        return {
            "title": self.title,
            "url": self.url,
            "source": self.source,
            "published": self.published.isoformat() if self.published else None,
        }


@dataclass
class Article:
    """Extracted content for one article.

    Attributes:
        url: Final publisher URL after redirect resolution.
        title: Page title, when the publisher provides one.
        content: Readable article text, truncated to the requested length.
        image_url: Lead image URL, or an empty string when none was found.
        truncated: Whether ``content`` was cut short.
    """

    url: str
    title: str
    content: str
    image_url: str = ""
    truncated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe primitives."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "image_url": self.image_url,
            "truncated": self.truncated,
        }


class GoogleNewsClient:
    """Async client for Google News RSS search, topic feeds, and article text.

    Use as an async context manager so the underlying session is closed::

        async with GoogleNewsClient(language="ko", region="KR") as client:
            items = await client.search("반도체")
    """

    def __init__(
        self,
        language: str = "en",
        region: str = "US",
        timeout: float = 10.0,
        max_concurrency: int = 5,
        max_retries: int = 3,
    ) -> None:
        """Initialize the client.

        Args:
            language: Google News ``hl`` code, e.g. ``en``, ``ko``, ``ja``.
            region: Google News ``gl`` code, e.g. ``US``, ``KR``, ``JP``.
            timeout: Per-request timeout in seconds.
            max_concurrency: Maximum simultaneous outbound HTTP requests.
            max_retries: Attempts per request before giving up.
        """
        self.language = language
        self.region = region
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._connection_limit = max(2 * max_concurrency, 10)

    async def __aenter__(self) -> "GoogleNewsClient":
        """Open the shared HTTP session."""
        # Trust certifi's bundle rather than whatever the host happens to have.
        # Many publishers serve incomplete chains that fail against a bare
        # system store, and a news reader that can't open news is useless.
        connector = aiohttp.TCPConnector(
            ssl=ssl.create_default_context(cafile=certifi.where()),
            limit=self._connection_limit,
            ttl_dns_cache=300,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Language": f"{self.language},en;q=0.8",
            },
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the shared HTTP session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self, query: str, max_results: int = 10, resolve_urls: bool = True
    ) -> List[NewsItem]:
        """Search Google News for a keyword.

        Args:
            query: Free-text search query.
            max_results: Maximum number of headlines to return.
            resolve_urls: Resolve each ``news.google.com`` link to the publisher
                URL. Costs two extra requests per item; turn it off for speed.

        Returns:
            Headlines, newest first as ordered by Google News.

        Raises:
            GoogleNewsError: If the feed cannot be fetched or parsed.
        """
        if not query or not query.strip():
            raise GoogleNewsError("query must not be empty")

        feed_url = (
            f"https://{GOOGLE_NEWS_HOST}/rss/search"
            f"?q={quote(query)}&{self._locale_params()}"
        )
        items = await self._fetch_feed(feed_url)
        return await self._finalize(items, max_results, resolve_urls)

    async def top_headlines(
        self, topic: str = "top", max_results: int = 10, resolve_urls: bool = True
    ) -> List[NewsItem]:
        """Fetch headlines for a Google News topic section.

        Args:
            topic: One of the keys of :data:`TOPIC_PATHS`.
            max_results: Maximum number of headlines to return.
            resolve_urls: Resolve each link to the publisher URL.

        Returns:
            Headlines for the requested topic.

        Raises:
            GoogleNewsError: If the topic is unknown, or the feed fails.
        """
        if topic not in TOPIC_PATHS:
            raise GoogleNewsError(
                f"unknown topic {topic!r}; expected one of {sorted(TOPIC_PATHS)}"
            )

        section = TOPIC_PATHS[topic]
        path = "/rss" if section is None else f"/rss/headlines/section/topic/{section}"
        feed_url = f"https://{GOOGLE_NEWS_HOST}{path}?{self._locale_params()}"
        items = await self._fetch_feed(feed_url)
        return await self._finalize(items, max_results, resolve_urls)

    async def read_article(self, url: str, max_length: int = 5000) -> Article:
        """Download one article and extract its text and lead image.

        Accepts either a publisher URL or a ``news.google.com`` redirect link.
        The page is fetched exactly once and parsed once for both outputs.

        Args:
            url: Article URL to read.
            max_length: Maximum characters of body text to return.

        Returns:
            The extracted article.

        Raises:
            GoogleNewsError: If the page cannot be fetched.
        """
        resolved = url
        if urlparse(url).netloc.endswith(GOOGLE_NEWS_HOST):
            resolved = await self._resolve_url(url)

        status, markup = await self._request("GET", resolved)
        if markup is None:
            raise GoogleNewsError(_fetch_failure_message(resolved, status))

        soup = BeautifulSoup(markup, "html.parser")
        image_url = self._extract_image(soup, resolved)
        title = self._extract_title(soup)
        body = self._extract_body(soup)

        truncated = len(body) > max_length
        if truncated:
            body = body[:max_length].rstrip() + "…"

        return Article(
            url=resolved,
            title=title,
            content=body,
            image_url=image_url,
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _locale_params(self) -> str:
        """Build the ``hl``/``gl``/``ceid`` query string for the current locale."""
        return (
            f"hl={self.language}&gl={self.region}"
            f"&ceid={self.region}:{self.language}"
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """Perform an HTTP request with concurrency limiting and backoff.

        Retries on 429 and 5xx with exponential backoff plus jitter, which is
        what Google's batchexecute endpoint returns under parallel load.

        Args:
            method: HTTP verb.
            url: Target URL.
            data: Form-encoded body for POST requests.
            headers: Extra request headers.

        Returns:
            ``(status, body)``. ``body`` is ``None`` when the request failed;
            ``status`` is the last HTTP status seen, or ``None`` if the request
            never got that far (DNS, TLS, timeout).

        Raises:
            GoogleNewsError: If the client is used outside its context manager.
        """
        if self._session is None:
            raise GoogleNewsError(
                "GoogleNewsClient must be used as an async context manager"
            )

        status: Optional[int] = None
        for attempt in range(self.max_retries):
            try:
                async with self._semaphore:
                    async with self._session.request(
                        method, url, data=data, headers=headers
                    ) as response:
                        status = response.status
                        if status == 429 or status >= 500:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=status,
                                message=f"HTTP {status}",
                            )
                        if status >= 400:
                            logger.warning("HTTP %s for %s", status, url)
                            return status, None
                        return status, await response.text()
            except (
                aiohttp.ClientConnectorCertificateError,
                aiohttp.ClientConnectorSSLError,
            ) as exc:
                # A bad certificate will be just as bad on the next attempt.
                logger.warning("TLS failure for %s: %s", url, exc)
                return status, None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == self.max_retries - 1:
                    logger.warning(
                        "giving up on %s after %s attempts: %s",
                        url,
                        self.max_retries,
                        exc,
                    )
                    return status, None
                delay = 0.5 * (2**attempt) + random.uniform(0, 0.3)
                logger.debug("retrying %s in %.2fs (%s)", url, delay, exc)
                await asyncio.sleep(delay)
        return status, None

    async def _fetch_feed(self, feed_url: str) -> List[NewsItem]:
        """Fetch and parse an RSS feed into headline items.

        Args:
            feed_url: Fully-formed Google News RSS URL.

        Returns:
            Every item in the feed, in feed order.

        Raises:
            GoogleNewsError: If the feed cannot be fetched or contains no items.
        """
        status, body = await self._request("GET", feed_url)
        if body is None:
            raise GoogleNewsError(_fetch_failure_message(feed_url, status))

        parsed = feedparser.parse(body)
        if parsed.bozo and not parsed.entries:
            raise GoogleNewsError(
                f"malformed feed from Google News: {parsed.bozo_exception}"
            )

        items: List[NewsItem] = []
        for entry in parsed.entries:
            title = clean_text(entry.get("title", ""))
            if not title:
                continue

            # Google News formats titles as "Headline - Publisher".
            source = (entry.get("source") or {}).get("title", "")
            if source:
                title = _strip_source_suffix(title, source)
            elif " - " in title:
                title, _, source = title.rpartition(" - ")
                title, source = title.strip(), source.strip()

            items.append(
                NewsItem(
                    title=title,
                    url=entry.get("link", ""),
                    source=source,
                    published=parse_date(entry.get("published", "")),
                )
            )
        return items

    async def _finalize(
        self, items: List[NewsItem], max_results: int, resolve_urls: bool
    ) -> List[NewsItem]:
        """Trim the feed to ``max_results`` and optionally resolve links.

        Args:
            items: Parsed feed items.
            max_results: Maximum number of items to keep.
            resolve_urls: Whether to resolve Google redirect links.

        Returns:
            The trimmed (and possibly resolved) items.
        """
        selected = [item for item in items if item.url][: max(1, max_results)]
        if not resolve_urls or not selected:
            return selected

        resolved = await asyncio.gather(
            *(self._resolve_url(item.url) for item in selected)
        )
        for item, url in zip(selected, resolved):
            item.url = url
        return selected

    async def _resolve_url(self, google_news_url: str) -> str:
        """Resolve a Google News redirect link to the publisher URL.

        Google encodes the destination in a ``c-wiz[data-p]`` attribute that has
        to be replayed against an internal batchexecute endpoint.

        Args:
            google_news_url: A ``news.google.com/rss/articles/...`` link.

        Returns:
            The publisher URL, or the input unchanged if resolution failed.
        """
        try:
            _, markup = await self._request("GET", google_news_url)
            if markup is None:
                return google_news_url

            element = BeautifulSoup(markup, "html.parser").select_one("c-wiz[data-p]")
            if element is None:
                logger.debug("no c-wiz[data-p] in %s", google_news_url)
                return google_news_url

            payload = json.loads(
                str(element["data-p"]).replace("%.@.", '["garturlreq",')
            )
            _, body = await self._request(
                "POST",
                BATCH_EXECUTE_URL,
                data={
                    "f.req": json.dumps(
                        [
                            [
                                [
                                    "Fbv4je",
                                    json.dumps(payload[:-6] + payload[-2:]),
                                    "null",
                                    "generic",
                                ]
                            ]
                        ]
                    )
                },
                headers={
                    "content-type": "application/x-www-form-urlencoded;charset=UTF-8"
                },
            )
            if body is None:
                return google_news_url

            envelope = json.loads(body.replace(")]}'", ""))[0][2]
            return json.loads(envelope)[1]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            logger.debug("could not resolve %s: %s", google_news_url, exc)
            return google_news_url

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Read the page title from Open Graph, then ``<title>``.

        Args:
            soup: Parsed article page.

        Returns:
            The title, or an empty string.
        """
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return clean_text(og_title["content"])
        if soup.title and soup.title.string:
            return clean_text(soup.title.string)
        return ""

    def _extract_body(self, soup: BeautifulSoup) -> str:
        """Extract readable article text from an already-parsed page.

        Strips page furniture, prefers a known article container, and falls back
        to the whole body.

        Args:
            soup: Parsed article page. Mutated in place.

        Returns:
            Cleaned plain-text article body.
        """
        for tag in soup(_BOILERPLATE_TAGS):
            tag.decompose()

        container = None
        for selector in _ARTICLE_SELECTORS:
            container = soup.select_one(selector)
            if container is not None:
                break
        if container is None:
            container = soup.body or soup

        converter = html2text.HTML2Text()
        converter.ignore_links = True
        converter.ignore_images = True
        converter.ignore_emphasis = True
        converter.body_width = 0
        return clean_text(converter.handle(str(container)))

    def _extract_image(self, soup: BeautifulSoup, base_url: str) -> str:
        """Find the lead image for an article.

        Tries Open Graph, Twitter cards, schema.org metadata, JSON-LD, then the
        first sufficiently large ``<img>``.

        Args:
            soup: Parsed article page.
            base_url: Article URL, used to absolutize relative image paths.

        Returns:
            An absolute image URL, or an empty string.
        """
        meta_candidates = (
            soup.find("meta", property="og:image"),
            soup.find("meta", attrs={"name": "twitter:image"}),
            soup.find("meta", attrs={"itemprop": "image"}),
        )
        for meta in meta_candidates:
            if meta and meta.get("content"):
                return urljoin(base_url, meta["content"].strip())

        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string or "")
            except (ValueError, TypeError):
                continue
            found = _image_from_json_ld(data)
            if found:
                return urljoin(base_url, found)

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
            if src and _is_plausible_lead_image(img, src):
                return urljoin(base_url, src.strip())

        return ""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _fetch_failure_message(url: str, status: Optional[int]) -> str:
    """Explain a failed fetch in terms the calling agent can act on.

    Args:
        url: The URL that could not be read.
        status: Last HTTP status seen, or ``None`` if the request never
            completed (DNS, TLS, or timeout failure).

    Returns:
        A message naming the likely cause and the useful next step.
    """
    if status in (401, 402, 403):
        return (
            f"publisher blocked automated access to {url} (HTTP {status}) — "
            "this is usually a paywall or bot protection. Try another result."
        )
    if status == 404:
        return f"article not found at {url} (HTTP 404) — the link may have expired."
    if status is not None:
        return f"could not fetch {url} (HTTP {status})."
    return (
        f"could not reach {url} — the request timed out or the host's TLS "
        "certificate could not be verified."
    )


def _image_from_json_ld(data: Any) -> Optional[str]:
    """Pull an image URL out of a JSON-LD blob.

    Args:
        data: Decoded JSON-LD, of any shape.

    Returns:
        The first image URL found, or ``None``.
    """
    if isinstance(data, list):
        for entry in data:
            found = _image_from_json_ld(entry)
            if found:
                return found
        return None

    if not isinstance(data, dict):
        return None

    image = data.get("image")
    if isinstance(image, str) and image:
        return image
    if isinstance(image, dict) and isinstance(image.get("url"), str):
        return image["url"]
    if isinstance(image, list):
        found = _image_from_json_ld({"image": image[0]}) if image else None
        if found:
            return found

    for key in ("mainEntity", "@graph"):
        if key in data:
            found = _image_from_json_ld(data[key])
            if found:
                return found
    return None


def _is_plausible_lead_image(img_tag: Any, src: str) -> bool:
    """Heuristically reject icons, trackers, and chrome.

    Args:
        img_tag: The ``<img>`` element.
        src: Its resolved source attribute.

    Returns:
        ``True`` if the image could plausibly be the article's lead image.
    """
    if src.startswith("data:") or len(src) < 12:
        return False

    lowered = src.lower()
    if any(pattern in lowered for pattern in _IMAGE_SKIP_PATTERNS):
        return False

    for dimension in ("width", "height"):
        raw = img_tag.get(dimension)
        if raw:
            try:
                if int(str(raw).rstrip("px")) < 200:
                    return False
            except ValueError:
                pass
    return True


def clean_text(text: str) -> str:
    """Normalize text without discarding meaningful characters.

    Unlike a character allow-list, this preserves ``&``, ``%``, ``$``, ``+`` and
    quotation marks, so figures and names such as "AT&T" or "up 5%" survive.

    Args:
        text: Raw text, possibly containing HTML entities.

    Returns:
        Entity-decoded, whitespace-normalized text.
    """
    if not text:
        return ""

    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse runs of blank lines to one, then runs of spaces/tabs to one.
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    return text.strip()


def parse_date(date_str: str) -> Optional[datetime]:
    """Parse an RSS date string into an aware ``datetime``.

    Args:
        date_str: Date string from a feed entry.

    Returns:
        A timezone-aware datetime, or ``None`` if it could not be parsed.
    """
    if not date_str:
        return None

    # feedparser normalizes to UTC and handles far more formats than strptime.
    # It lives in a private module, so fall through to strptime if it moves.
    parsed = None
    if _feedparser_parse_date is not None:
        try:
            parsed = _feedparser_parse_date(date_str)
        except Exception:  # noqa: BLE001 - third-party parser, any failure is fine
            parsed = None
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except (TypeError, ValueError):
            pass

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            value = datetime.strptime(date_str, fmt)
        except ValueError:
            continue
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None
