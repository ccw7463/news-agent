"""News agent: search Google News over MCP, read what matters, summarize.

The graph is four nodes:

    call_model → tools → read_articles → summarize → answer

``tools`` returns headlines only. ``read_articles`` then spends fetches on a
bounded slice of them, which is the whole reason the news server splits search
from reading — a search that also scraped every body pushed roughly 8k tokens
into the context whether or not any of it was relevant.
"""

import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

import pytz
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.modules.articles import collect_articles, fetch_bodies
from src.modules.llm import build_model, model_label
from src.modules.mcp_client import news_tools

load_dotenv()

console = Console()
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "WARNING").upper())

KST = pytz.timezone("Asia/Seoul")

DEFAULT_QUESTION = "What is the latest news about AI?"
# Bodies cost a fetch each and dominate both latency and tokens, so the default
# reads far fewer articles than a search returns.
DEFAULT_MAX_READ = int(os.environ.get("MAX_READ_ARTICLES", "5"))
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_ARTICLE_LENGTH", "4000"))

# read_article declares Field(ge=200, le=100_000). A value outside that fails
# schema validation on every call, and each failure is swallowed into
# `read_error` — so the run silently degrades to headlines with no hint that the
# flag caused it.
MIN_ARTICLE_LENGTH = 200
MAX_ARTICLE_LENGTH = 100_000

# The hosted server answers in en/US unless a call says otherwise, so a Korean
# question searched with the defaults returns nothing at all. The tools take
# language and region per call; the model has to actually pass them.
SEARCH_SYSTEM_PROMPT = (
    "You search Google News to answer news questions.\n\n"
    "Set `language` and `region` to match the language the user wrote in — "
    "for example language='ko', region='KR' for a Korean question, "
    "language='ja', region='JP' for Japanese. Leave them unset only for "
    "English.\n\n"
    "Keep `query` to a few plain keywords. Do not add years, dates, or words "
    "like 'latest' or 'news'; the feed is already current and those only "
    "narrow it to nothing."
)

MODE_BLURB = {
    "smithery": "Smithery gateway (published listing @ccw7463/google-rss-mcp)",
    "http": "origin instance over HTTP (no API key)",
    "stdio": "upstream source via uvx",
    "local": "vendored fallback in src/modules/mcp_servers",
}


class NewsAgentState(MessagesState):
    """Graph state: the message history plus the articles gathered so far."""

    articles: List[Dict[str, Any]]


def _find_tool(tools: List[Any], name: str) -> Any:
    """Look up one tool by name.

    Args:
        tools: Tools loaded from the server.
        name: Tool name to find.

    Returns:
        The tool.

    Raises:
        RuntimeError: If the server did not expose it, which means the contract
            has drifted and the graph cannot run as written.
    """
    for tool in tools:
        if tool.name == name:
            return tool
    available = ", ".join(t.name for t in tools) or "none"
    raise RuntimeError(f"server exposes no {name!r} tool (available: {available})")


def build_graph(model, tools: List[Any], max_read: int, max_length: int):
    """Wire the agent graph.

    Args:
        model: Chat model to bind tools to and summarize with.
        tools: Tools loaded from the news server.
        max_read: Maximum article bodies to fetch per run.
        max_length: Characters of body text to request per article.

    Returns:
        The compiled graph.
    """
    read_article = _find_tool(tools, "read_article")
    # The model picks headlines; it must not spend the run reading every result
    # one call at a time, so reading is done by a node instead.
    search_tools = [t for t in tools if t.name != "read_article"]

    def call_model(state: NewsAgentState) -> Dict[str, Any]:
        """Let the model choose a search tool for the question.

        The search is required, not offered. Asked "what is the latest news
        about X?", a model with a strong prior will happily answer from memory
        and never call a tool — the same fabricated-briefing failure that
        `answer_node` guards against, arriving by a route that skips it.
        """
        messages = [SystemMessage(content=SEARCH_SYSTEM_PROMPT), *state["messages"]]
        response = model.bind_tools(search_tools, tool_choice="any").invoke(messages)
        return {"messages": response}

    def ungrounded_node(state: NewsAgentState) -> Dict[str, Any]:
        """Refuse to pass off an unsearched answer as news.

        Reached only when the model ignored `tool_choice="any"`. Whatever it
        wrote instead came from its training data, so it is dropped rather than
        printed under a news heading.
        """
        return {
            "messages": [
                AIMessage(
                    content=(
                        "No news search was run for this question, so there is "
                        "nothing retrieved to report. Try phrasing it as a news "
                        "topic — a subject, a company, or an event."
                    )
                )
            ]
        }

    async def read_articles_node(state: NewsAgentState) -> Dict[str, Any]:
        """Turn headlines into readable articles."""
        headlines = collect_articles(
            msg for msg in state["messages"] if isinstance(msg, ToolMessage)
        )
        if not headlines:
            console.print(
                Panel(
                    "[yellow]The search returned no headlines.[/yellow]",
                    border_style="yellow",
                )
            )
            return {"articles": []}

        planned = min(max_read, len(headlines))
        console.print(
            Panel(
                f"[bold yellow]📰 {len(headlines)} headlines found — "
                f"reading the top {planned}[/bold yellow]",
                border_style="yellow",
            )
        )

        def report(index: int, article: Dict[str, Any], error: str) -> None:
            title = (article.get("title") or "")[:64]
            if error:
                console.print(f"[red]   ✗ {title} — {error}[/red]")
            else:
                console.print(f"[green]   ✓ {title}[/green]")

        readable = await fetch_bodies(
            headlines,
            read_article,
            limit=planned,
            max_length=max_length,
            on_result=report,
        )
        console.print(
            Panel(
                f"[bold green]✅ Read {len(readable)}/{planned} articles[/bold green]",
                border_style="green",
            )
        )
        # Headlines without a body still carry a title, source and timestamp,
        # which is enough for the final answer to mention them.
        return {"articles": headlines}

    async def summarize_node(state: NewsAgentState) -> Dict[str, Any]:
        """Summarize each article that has a body, in parallel."""
        articles = state.get("articles", [])
        readable = [a for a in articles if a.get("content")]
        if not readable:
            return {"articles": articles}

        user_query = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
            "Summarize the articles.",
        )

        console.print(
            Panel(
                f"[bold yellow]🤖 Summarizing {len(readable)} articles...[/bold yellow]",
                border_style="yellow",
            )
        )

        async def summarize(article: Dict[str, Any]) -> bool:
            prompt = [
                SystemMessage(
                    content=(
                        "You are a news summarization expert. Summarize the "
                        "article in five sentences or fewer, answering the "
                        "user's question. Use only the article text; if it does "
                        "not address the question, say so."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question: {user_query}\n"
                        f"Title: {article.get('title', '')}\n"
                        f"Source: {article.get('source', 'unknown')}\n"
                        f"Article:\n{article.get('content', '')}\n\nSummary:"
                    )
                ),
            ]
            try:
                result = await model.ainvoke(prompt)
                article["summary"] = result.content.strip()
                return True
            except Exception as exc:  # noqa: BLE001 - one failure must not sink the run
                console.print(f"[red]   ✗ summary failed: {exc}[/red]")
                article["summary"] = ""
                return False

        results = await asyncio.gather(
            *(summarize(a) for a in readable), return_exceptions=True
        )
        ok = sum(1 for r in results if r is True)
        console.print(
            Panel(
                f"[bold green]✅ Summarized {ok}/{len(readable)}[/bold green]",
                border_style="green",
            )
        )
        return {"articles": articles}

    def answer_node(state: NewsAgentState) -> Dict[str, Any]:
        """Fuse the per-article summaries into one answer."""
        articles = state.get("articles", [])
        summarized = [a for a in articles if a.get("summary")]

        user_query = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
            "Summarize the news.",
        )

        # With nothing retrieved there is nothing to synthesize, and asking the
        # model to answer anyway is an invitation to answer from memory. It will
        # take it: a search returning zero results once produced a confident
        # briefing citing seven newspapers, none of which had been fetched. For a
        # news agent that failure is worse than no answer, so it never reaches
        # the model.
        if not articles:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "The news search returned no results for this "
                            "question, so there is nothing to report. Try "
                            "different keywords, or ask in the language the "
                            "coverage is published in."
                        )
                    )
                ]
            }

        if not summarized:
            headlines = "\n".join(
                f"- {a.get('title', '')} ({a.get('source', 'unknown')})"
                for a in articles[:20]
            )
            # "nothing was fetched" and "fetching worked, summarizing failed"
            # are different failures, and saying the wrong one misdirects
            # whoever reads the answer.
            reason = (
                "The articles were fetched but could not be summarized"
                if any(a.get("content") for a in articles)
                else "No article bodies could be read"
            )
            body = (
                f"{reason}. Answer from these headlines alone, use nothing "
                f"else, and say so.\n\n"
                f"Question: {user_query}\nHeadlines:\n{headlines}"
            )
        else:
            blocks = "\n\n".join(
                f"[{i}] {a.get('title', '')} — {a.get('source', 'unknown')}"
                f" ({a.get('published') or 'undated'})\n{a['summary']}"
                for i, a in enumerate(summarized, 1)
            )
            body = (
                f"Question: {user_query}\n\nArticle summaries:\n{blocks}\n\n"
                "Write the final answer. Cite sources by their bracketed number."
            )

        prompt = [
            SystemMessage(
                content=(
                    "You synthesize retrieved news into one answer to the "
                    "user's question. Use only what is given below — never your "
                    "own recollection of the news, and never a source that does "
                    "not appear here. Attribute every claim to its source, and "
                    "answer in the language the user asked in."
                )
            ),
            HumanMessage(content=body),
        ]
        return {"messages": [model.invoke(prompt)]}

    builder = StateGraph(NewsAgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("ungrounded", ungrounded_node)
    builder.add_node("tools", ToolNode(search_tools))
    builder.add_node("read_articles", read_articles_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("answer", answer_node)

    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", tools_condition, {"tools": "tools", END: "ungrounded"}
    )
    builder.add_edge("ungrounded", END)
    builder.add_edge("tools", "read_articles")
    builder.add_edge("read_articles", "summarize")
    builder.add_edge("summarize", "answer")
    builder.add_edge("answer", END)
    return builder.compile()


def _print_sources(articles: List[Dict[str, Any]]) -> None:
    """Show what the answer was built from, so it can be checked."""
    cited = [a for a in articles if a.get("summary")]
    if not cited:
        return
    table = Table(title="📎 Sources", show_header=True, header_style="bold white")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan")
    table.add_column("Source", style="white", no_wrap=True)
    table.add_column("Published", style="dim", no_wrap=True)
    for i, article in enumerate(cited, 1):
        table.add_row(
            str(i),
            (article.get("title") or "")[:70],
            article.get("source") or "unknown",
            (article.get("published") or "")[:16],
        )
    console.print(table)


async def main() -> None:
    """Run one question through the agent."""
    parser = argparse.ArgumentParser(description="Ask the news agent a question.")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--max-read",
        type=int,
        default=DEFAULT_MAX_READ,
        help="Maximum article bodies to fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Characters of body text per article (default: %(default)s).",
    )
    args = parser.parse_args()

    clamped_length = max(MIN_ARTICLE_LENGTH, min(MAX_ARTICLE_LENGTH, args.max_length))
    if clamped_length != args.max_length:
        console.print(
            f"[yellow]--max-length {args.max_length} is outside the server's "
            f"{MIN_ARTICLE_LENGTH}–{MAX_ARTICLE_LENGTH} range; using "
            f"{clamped_length}.[/yellow]"
        )
        args.max_length = clamped_length

    if args.max_read < 0:
        console.print(
            f"[yellow]--max-read {args.max_read} is negative; using 0.[/yellow]"
        )
        args.max_read = 0

    started = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")
    console.print(
        Panel.fit(
            "[bold blue]🚀 News Agent — LangGraph × google-rss-mcp[/bold blue]\n"
            f"[dim]{model_label()} via OpenRouter · {started} KST[/dim]",
            border_style="blue",
        )
    )

    try:
        model = build_model()
    except RuntimeError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
        return

    def on_attempt(mode: str, status: str, detail: str) -> None:
        if status == "trying":
            console.print(f"[dim]📡 trying {mode} — {MODE_BLURB.get(mode, mode)}[/dim]")
        elif status == "failed":
            console.print(f"[yellow]   ✗ {mode}: {detail}[/yellow]")

    # One session for the whole run: tool calls reuse it instead of paying a
    # fresh MCP handshake (and, in the stdio modes, a new subprocess) each time.
    try:
        async with news_tools(on_attempt=on_attempt) as (tools, mode):
            await _run(model, tools, mode, args)
    except RuntimeError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))


async def _run(model, tools, mode: str, args) -> None:
    """Build the graph and answer one question over an open session.

    Args:
        model: The chat model.
        tools: Tools bound to the live session.
        mode: Name of the connection mode that answered, for display.
        args: Parsed command-line arguments.
    """
    console.print(
        Panel(
            f"[bold green]✅ Connected via {mode}[/bold green]\n"
            f"[dim]{MODE_BLURB.get(mode, mode)}[/dim]",
            border_style="green",
        )
    )

    tools_table = Table(title="🔧 Tools", show_header=True, header_style="bold white")
    tools_table.add_column("Name", style="cyan", no_wrap=True)
    tools_table.add_column("Description", style="white")
    for tool in tools:
        tools_table.add_row(tool.name, (tool.description or "").split("\n")[0])
    console.print(tools_table)

    try:
        graph = build_graph(model, tools, args.max_read, args.max_length)
    except RuntimeError as exc:
        console.print(Panel(f"[bold red]{exc}[/bold red]", border_style="red"))
        return

    console.print(
        Panel(
            f"[bold magenta]🔍 {args.question}[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )

    try:
        response = await graph.ainvoke(
            {"messages": [HumanMessage(content=args.question)], "articles": []}
        )
    # Surface the failure as a message, not a traceback at the user.
    except Exception as exc:  # noqa: BLE001
        console.print(
            Panel(f"[bold red]❌ Run failed: {exc}[/bold red]", border_style="red")
        )
        return

    messages = response.get("messages", [])
    answer = messages[-1].content if messages else ""
    console.print(
        Panel(
            answer or "[red]No answer produced.[/red]",
            title="[bold magenta]📋 Answer[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )
    _print_sources(response.get("articles", []))


if __name__ == "__main__":
    asyncio.run(main())
