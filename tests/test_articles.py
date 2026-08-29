"""Tests for reading the news server's tool output back into article records.

This is the seam that broke when the server split search from reading: the
search tools used to return a JSON *list* of articles with bodies attached, and
now return a ``{count, articles}`` envelope of headlines with no body at all.
Anything that silently returns an empty list here produces a confident,
sourceless answer, so these cases are pinned.
"""

import copy
import json

import pytest

from src.modules.articles import collect_articles, extract_headlines, fetch_bodies


class FakeToolMessage:
    """Stand-in for a LangChain ``ToolMessage``."""

    def __init__(self, content):
        self.content = content


SEARCH_PAYLOAD = {
    "query": "AI",
    "language": "en",
    "region": "US",
    "count": 2,
    "articles": [
        {
            "title": "AT&T stock rose 5% to $120",
            "url": "https://example.com/a",
            "source": "Example News",
            "published": "2026-08-29T09:00:00+00:00",
        },
        {
            "title": "Second story",
            "url": "https://example.com/b",
            "source": "Other News",
            "published": None,
        },
    ],
}


def test_extracts_articles_from_the_envelope():
    """The current search/headline shape is a dict, not a list."""
    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    assert [a["url"] for a in articles] == [
        "https://example.com/a",
        "https://example.com/b",
    ]


def test_envelope_arrives_as_a_json_string():
    """Tool content reaches the graph as text, not as a decoded object."""
    msg = FakeToolMessage(json.dumps(SEARCH_PAYLOAD))
    assert len(collect_articles([msg])) == 2


def test_content_blocks_are_unwrapped():
    """Some adapter versions wrap the result in a list of content blocks."""
    msg = FakeToolMessage([{"type": "text", "text": json.dumps(SEARCH_PAYLOAD)}])
    assert len(collect_articles([msg])) == 2


def test_read_article_result_is_kept_whole():
    """A direct read_article call already carries a body; do not drop it."""
    payload = {
        "url": "https://example.com/a",
        "title": "A",
        "content": "body text",
        "image_url": "",
        "truncated": False,
    }
    articles = extract_headlines(payload)
    assert len(articles) == 1
    assert articles[0]["content"] == "body text"


def test_duplicate_urls_collapse():
    """A search and a topic call overlapping must not summarize twice."""
    msgs = [
        FakeToolMessage(json.dumps(SEARCH_PAYLOAD)),
        FakeToolMessage(json.dumps(SEARCH_PAYLOAD)),
    ]
    assert len(collect_articles(msgs)) == 2


def test_non_json_content_is_skipped_not_fatal():
    """An error string from the server must not take the run down."""
    msgs = [
        FakeToolMessage("publisher blocked automated access (HTTP 403)"),
        FakeToolMessage(json.dumps(SEARCH_PAYLOAD)),
    ]
    assert len(collect_articles(msgs)) == 2


def test_titles_keep_ampersands_and_currency():
    """Regression guard for the character allow-list bug upstream fixed."""
    articles = collect_articles([FakeToolMessage(json.dumps(SEARCH_PAYLOAD))])
    assert articles[0]["title"] == "AT&T stock rose 5% to $120"


class FakeReadArticle:
    """Records calls and returns a canned body, or raises for one URL."""

    def __init__(self, fail_for=()):
        self.calls = []
        self.fail_for = set(fail_for)

    async def ainvoke(self, payload):
        self.calls.append(payload)
        url = payload["url"]
        if url in self.fail_for:
            raise RuntimeError("publisher blocked automated access (HTTP 403)")
        return json.dumps(
            {
                "url": url,
                "title": "T",
                "content": f"body of {url}",
                "image_url": "",
                "truncated": False,
            }
        )


@pytest.mark.asyncio
async def test_fetch_bodies_respects_the_limit():
    """Reading every headline would undo the point of the search/read split."""
    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    reader = FakeReadArticle()
    readable = await fetch_bodies(articles, reader, limit=1)
    assert len(reader.calls) == 1
    assert len(readable) == 1


@pytest.mark.asyncio
async def test_a_paywalled_article_does_not_sink_the_others():
    """403s are expected; the rest of the batch must still come back."""
    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    reader = FakeReadArticle(fail_for={"https://example.com/a"})
    readable = await fetch_bodies(articles, reader, limit=2)
    assert [a["url"] for a in readable] == ["https://example.com/b"]
    assert "403" in articles[0]["read_error"]


@pytest.mark.asyncio
async def test_already_read_articles_are_not_refetched():
    """A body from a direct read_article call is not worth paying for twice."""
    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    articles[0]["content"] = "already have it"
    reader = FakeReadArticle()
    await fetch_bodies(articles, reader, limit=5)
    assert [c["url"] for c in reader.calls] == ["https://example.com/b"]


def test_adapter_error_prose_is_stripped_to_the_servers_message():
    """The status code and paywall hint are the useful part; keep them."""
    from src.modules.articles import _coerce_text

    raw = [
        {
            "type": "text",
            "text": (
                "Internal error: Error calling tool 'read_article': publisher "
                "blocked automated access to https://example.com/a (HTTP 403) "
                "— this is usually a paywall or bot protection."
            ),
            "id": "lc_abc",
        }
    ]
    text = _coerce_text(raw)
    assert text.startswith("publisher blocked automated access")
    assert "HTTP 403" in text
    assert "Error calling tool" not in text


@pytest.mark.asyncio
async def test_a_failed_read_keeps_the_servers_explanation():
    """A generic 'no content' message would re-hide what the server told us."""

    class BlockedReader:
        async def ainvoke(self, payload):
            return [
                {
                    "type": "text",
                    "text": (
                        "Internal error: Error calling tool 'read_article': "
                        "publisher blocked automated access (HTTP 403) — this "
                        "is usually a paywall or bot protection."
                    ),
                }
            ]

    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    readable = await fetch_bodies(articles, BlockedReader(), limit=1)
    assert readable == []
    assert "HTTP 403" in articles[0]["read_error"]
    assert "Error calling tool" not in articles[0]["read_error"]


def test_live_content_block_shape_is_parsed():
    """The exact shape the adapter returns, including its extra id field."""
    raw = [
        {
            "type": "text",
            "text": json.dumps(SEARCH_PAYLOAD),
            "id": "lc_988309ff",
        }
    ]
    assert len(collect_articles([FakeToolMessage(raw)])) == 2


def test_the_publisher_suffix_is_stripped_from_titles():
    """Google appends " - Publisher" to every title; the server fails to strip it."""
    payload = {
        "count": 1,
        "articles": [
            {
                "title": "빌 게이츠 “AI 규제 필요” - 뉴시스",
                "url": "https://example.com/a",
                "source": "뉴시스",
                "published": None,
            }
        ],
    }
    article = collect_articles([FakeToolMessage(json.dumps(payload))])[0]
    assert article["title"] == "빌 게이츠 “AI 규제 필요”"
    assert article["source"] == "뉴시스"


def test_a_title_that_is_only_the_suffix_is_left_alone():
    """Stripping must never empty a headline."""
    payload = {
        "count": 1,
        "articles": [
            {"title": " - 뉴시스", "url": "https://example.com/a", "source": "뉴시스"}
        ],
    }
    assert collect_articles([FakeToolMessage(json.dumps(payload))])[0]["title"] == (
        " - 뉴시스"
    )


def test_an_unrelated_dash_is_not_treated_as_a_suffix():
    payload = {
        "count": 1,
        "articles": [
            {
                "title": "EU - US trade talks stall",
                "url": "https://example.com/a",
                "source": "Reuters",
            }
        ],
    }
    assert collect_articles([FakeToolMessage(json.dumps(payload))])[0]["title"] == (
        "EU - US trade talks stall"
    )


@pytest.mark.asyncio
async def test_an_empty_body_is_not_reported_as_raw_json():
    """A well-formed result with no text would otherwise print the whole document."""

    class EmptyBodyReader:
        async def ainvoke(self, payload):
            return json.dumps(
                {
                    "url": payload["url"],
                    "title": "Google News",
                    "content": "",
                    "image_url": "",
                    "truncated": False,
                }
            )

    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    readable = await fetch_bodies(articles, EmptyBodyReader(), limit=1)
    assert readable == []
    assert articles[0]["read_error"] == "the page had no extractable article text"
    assert "{" not in articles[0]["read_error"]


@pytest.mark.asyncio
async def test_a_negative_limit_reads_nothing_rather_than_everything():
    """[:-1] would fetch all but one — the opposite of a budget."""
    articles = extract_headlines(copy.deepcopy(SEARCH_PAYLOAD))
    reader = FakeReadArticle()
    await fetch_bodies(articles, reader, limit=-1)
    assert reader.calls == []


def test_a_doubled_publisher_suffix_is_fully_stripped():
    """Seen live: the publisher's own title ends with its name, and Google appends it again."""
    payload = {
        "count": 1,
        "articles": [
            {
                "title": '"삼전닉스, 美 수출규제 대비" - 머니투데이 - 머니투데이',
                "url": "https://example.com/a",
                "source": "머니투데이",
            }
        ],
    }
    article = collect_articles([FakeToolMessage(json.dumps(payload))])[0]
    assert article["title"] == '"삼전닉스, 美 수출규제 대비"'
