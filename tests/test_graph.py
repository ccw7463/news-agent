"""End-to-end wiring test for the agent graph, with no network and no LLM.

The graph changed shape when the server split search from reading, so what is
worth pinning here is that a tool call actually flows through the new
``read_articles`` node and reaches the answer with sources attached.
"""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from main import build_graph

SEARCH_RESULT = {
    "query": "AI",
    "language": "en",
    "region": "US",
    "count": 2,
    "articles": [
        {
            "title": "AT&T bets $5B on AI",
            "url": "https://example.com/a",
            "source": "Example News",
            "published": "2026-08-29T09:00:00+00:00",
        },
        {
            "title": "Paywalled scoop",
            "url": "https://example.com/blocked",
            "source": "Walled Times",
            "published": "2026-08-29T08:00:00+00:00",
        },
    ],
}


def _tools():
    """Build stand-ins for the three server tools."""

    async def search_news(query: str, max_results: int = 10) -> str:
        return json.dumps(SEARCH_RESULT)

    async def read_article(url: str, max_length: int = 5000) -> str:
        if "blocked" in url:
            return (
                "Internal error: Error calling tool 'read_article': publisher "
                "blocked automated access (HTTP 403) — this is usually a "
                "paywall or bot protection."
            )
        return json.dumps(
            {
                "url": url,
                "title": "AT&T bets $5B on AI",
                "content": "AT&T said it would spend $5B, up 5%, on AI.",
                "image_url": "",
                "truncated": False,
            }
        )

    return [
        StructuredTool.from_function(
            coroutine=search_news, name="search_news", description="Search."
        ),
        StructuredTool.from_function(
            coroutine=read_article, name="read_article", description="Read."
        ),
    ]


class StubModel:
    """A model that always searches once, then echoes what it was given."""

    def __init__(self):
        self.summary_calls = 0

    def bind_tools(self, tools, **kwargs):
        assert kwargs.get("tool_choice") == "any", (
            "the search must be required; a model left free to skip it will "
            "answer news questions from memory"
        )
        assert all(t.name != "read_article" for t in tools), (
            "read_article must not be bound to the model; the graph reads "
            "articles itself so the model cannot burn the run one call at a time"
        )
        return self

    def invoke(self, messages):
        if isinstance(messages, list) and any(
            isinstance(m, HumanMessage) and "Article summaries" in str(m.content)
            for m in messages
        ):
            return AIMessage(content="FINAL: AT&T is spending on AI [1].")
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "search_news", "args": {"query": "AI"}, "id": "call_1"}
            ],
        )

    async def ainvoke(self, prompt):
        self.summary_calls += 1
        return AIMessage(content="AT&T committed $5B to AI, a 5% increase.")


@pytest.mark.asyncio
async def test_a_question_flows_through_search_read_and_answer():
    model = StubModel()
    graph = build_graph(model, _tools(), max_read=2, max_length=1000)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="AI news?")], "articles": []}
    )

    assert result["messages"][-1].content.startswith("FINAL:")
    articles = result["articles"]
    assert len(articles) == 2
    # Only the readable one is summarized; the paywalled one keeps its reason.
    assert articles[0]["summary"]
    assert "summary" not in articles[1] or not articles[1].get("summary")
    assert "HTTP 403" in articles[1]["read_error"]
    assert model.summary_calls == 1


@pytest.mark.asyncio
async def test_currency_and_ampersands_survive_the_whole_pipeline():
    """Regression guard: the upstream text-cleaning bug corrupted these."""
    model = StubModel()
    graph = build_graph(model, _tools(), max_read=1, max_length=1000)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="AI news?")], "articles": []}
    )
    assert result["articles"][0]["title"] == "AT&T bets $5B on AI"
    assert "$5B" in result["articles"][0]["content"]


@pytest.mark.asyncio
async def test_max_read_bounds_the_fetches():
    """The read budget is the whole point of the search/read split."""
    model = StubModel()
    graph = build_graph(model, _tools(), max_read=1, max_length=1000)
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="AI news?")], "articles": []}
    )
    read = [a for a in result["articles"] if a.get("content") or a.get("read_error")]
    assert len(read) == 1


def test_a_server_without_read_article_fails_loudly():
    """A drifted contract must not degrade into silent headline-only answers."""
    only_search = [t for t in _tools() if t.name == "search_news"]
    with pytest.raises(RuntimeError, match="read_article"):
        build_graph(StubModel(), only_search, max_read=1, max_length=1000)


class EmptySearchTools:
    """A server whose search legitimately finds nothing."""

    @staticmethod
    def build():
        async def search_news(query: str, max_results: int = 10) -> str:
            return json.dumps(
                {
                    "query": query,
                    "language": "en",
                    "region": "US",
                    "count": 0,
                    "articles": [],
                }
            )

        async def read_article(url: str, max_length: int = 5000) -> str:
            raise AssertionError("nothing was found; nothing should be read")

        return [
            StructuredTool.from_function(
                coroutine=search_news, name="search_news", description="Search."
            ),
            StructuredTool.from_function(
                coroutine=read_article, name="read_article", description="Read."
            ),
        ]


class CountingModel(StubModel):
    """Records how many times the model was asked to write prose."""

    def __init__(self):
        super().__init__()
        self.prose_calls = 0

    def invoke(self, messages):
        result = super().invoke(messages)
        if not result.tool_calls:
            self.prose_calls += 1
        return result


@pytest.mark.asyncio
async def test_an_empty_search_never_reaches_the_model():
    """Asked to answer from nothing, the model answers from memory instead.

    A real run once returned zero headlines and produced a confident briefing
    citing seven newspapers, none of which had been fetched. For a news agent
    that is worse than no answer, so an empty result must not reach the model.
    """
    model = CountingModel()
    graph = build_graph(model, EmptySearchTools.build(), max_read=3, max_length=1000)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="AI 규제 뉴스")], "articles": []}
    )

    assert result["articles"] == []
    assert model.prose_calls == 0, "the model was asked to write from nothing"
    assert model.summary_calls == 0
    answer = result["messages"][-1].content
    assert "no results" in answer.lower()


@pytest.mark.asyncio
async def test_the_search_prompt_tells_the_model_to_match_the_locale():
    """The server defaults to en/US, so a Korean question needs ko/KR passed."""
    from main import SEARCH_SYSTEM_PROMPT

    assert "language" in SEARCH_SYSTEM_PROMPT and "region" in SEARCH_SYSTEM_PROMPT
    assert "ko" in SEARCH_SYSTEM_PROMPT and "KR" in SEARCH_SYSTEM_PROMPT


def test_the_search_prompt_is_actually_sent(monkeypatch):
    """A prompt that never reaches the model is not a fix."""
    seen = {}

    class Recorder(StubModel):
        def invoke(self, messages):
            seen.setdefault("first", messages)
            return super().invoke(messages)

    from langchain_core.messages import SystemMessage

    model = Recorder()
    graph = build_graph(model, _tools(), max_read=1, max_length=500)
    asyncio.run(
        graph.ainvoke({"messages": [HumanMessage(content="hi")], "articles": []})
    )
    assert isinstance(seen["first"][0], SystemMessage)


class RefusingModel(StubModel):
    """A model that answers from memory instead of calling the search."""

    def invoke(self, messages):
        if isinstance(messages, list) and any(
            isinstance(m, HumanMessage) and "Article summaries" in str(m.content)
            for m in messages
        ):
            return AIMessage(content="FINAL: unreachable")
        return AIMessage(content="I already know: AT&T announced a $5B AI fund.")


@pytest.mark.asyncio
async def test_an_unsearched_answer_is_discarded():
    """`tool_choice="any"` is a request, not a guarantee; the graph enforces it.

    Without this the model can answer a news question straight from training
    data and the run prints it under a news heading — the same fabrication
    `test_an_empty_search_never_reaches_the_model` guards, arriving by the route
    that skips `answer_node` entirely.
    """
    model = RefusingModel()
    graph = build_graph(model, _tools(), max_read=2, max_length=1000)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="AI news?")], "articles": []}
    )

    answer = result["messages"][-1].content
    assert "AT&T announced a $5B AI fund" not in answer
    assert "No news search was run" in answer
