# news-agent

A LangGraph agent over [google-rss-mcp](https://github.com/ccw7463/google-rss-mcp),
a Google News MCP server. It searches headlines, reads only the articles worth
reading, and answers with sources. The LLM is served through OpenRouter.

```
call_model ──▶ tools ──▶ read_articles ──▶ summarize ──▶ answer
              headlines   bounded fetch     per article   fused, cited
```

The split matters. The search tools return **headlines only** — title, publisher
URL, source, timestamp. Bodies come from a separate `read_article` call, so the
agent spends fetches on a handful of results instead of scraping every hit into
the context.

## Connecting to the server

The published server is reachable four ways. Set `MCP_MODE` to pin one, or leave
it unset to walk the chain from wherever your credentials allow.

| `MCP_MODE` | What it is | Needs |
| --- | --- | --- |
| `smithery` | The Smithery gateway, proxying the listing at [@ccw7463/google-rss-mcp](https://smithery.ai/server/@ccw7463/google-rss-mcp) | `SMITHERY_API_KEY` |
| `http` | The origin instance the gateway proxies to | nothing |
| `stdio` | Upstream source, run locally by `uvx` | `uv`, network to GitHub once |
| `local` | The vendored copy in `src/modules/mcp_servers` | nothing at all |

Unset, the chain is `smithery → http → stdio → local` when a Smithery key is
present and `http → stdio → local` when it is not.

**A Smithery key is optional.** The gateway is an OAuth 2.0 protected resource —
it advertises `WWW-Authenticate: Bearer` and answers **401** to anonymous
requests on every one of its hostnames (`…run.tools`, `server.smithery.ai`,
`mcp.smithery.ai`), so there is no keyless path through it. But that is
Smithery's access control in front of the server, not something the server
requires: `http` mode reaches the very same instance the gateway proxies to,
with no key and no account.

The gateway URL is the registry's `deploymentUrl` verbatim, with **no `/mcp`
path** — `/mcp` on that host answers 404:

```bash
curl https://registry.smithery.ai/servers/@ccw7463/google-rss-mcp
```

Setting `MCP_MODE` explicitly disables the fallback, on purpose: a deployment
that must prove it is talking to Smithery should fail loudly rather than quietly
serve from the vendored copy.

All four modes expose the same three tools with the same schemas, so nothing
downstream depends on which one answered.

| Tool | Returns |
| --- | --- |
| `search_news` | Headlines matching a keyword |
| `get_top_headlines` | Headlines for a topic section |
| `read_article` | One article's text, title, and lead image |

### Locale

The hosted instances answer in `en` / `US` and take `language` / `region` per
tool call, so every caller can ask for their own. `GOOGLE_RSS_LANGUAGE` and
`GOOGLE_RSS_REGION` apply only to the `stdio` and `local` modes, which this repo
launches itself.

## Getting started

```bash
uv sync                           # creates .venv from uv.lock
cp .env.example .env              # then fill in OPENROUTER_API_KEY
uv run python main.py "What is the latest news about AI?"
```

Options:

```bash
uv run python main.py "반도체 뉴스" --max-read 3 --max-length 3000
```

`--max-read` caps how many article bodies are fetched (default 5); `--max-length`
caps characters per body (default 4000). Both are also settable as
`MAX_READ_ARTICLES` and `MAX_ARTICLE_LENGTH`.

## The model

Everything goes through [OpenRouter](https://openrouter.ai), which speaks the
OpenAI wire format — so `langchain-openai` is the client, and the only thing
that makes it OpenRouter is the base URL. Swapping models is one environment
variable:

```bash
OPENROUTER_MODEL=google/gemini-3-flash-preview   # the default
```

Any OpenRouter model that supports **tool calling** works; the agent needs it to
choose a search. One run costs `1 + N + 1` completions — one to pick the search,
one per article read, one to fuse the answer — so seven calls at the default
`--max-read 5`.

The default is a *preview* model and may be withdrawn; `google/gemini-3.7-flash`
and `google/gemini-3.1-flash-lite` are stable alternatives.

## Tests

```bash
uv run pytest
```

Verified on CPython 3.11 and 3.14.

40 tests, no network and no LLM calls. They cover the parts that actually broke
when the server was refactored: the tool-result shapes, the connection fallback
chain, and the graph flowing a search through to a cited answer.

## Layout

```
main.py                         graph, CLI, output
src/modules/mcp_client.py       connection modes and fallback chain
src/modules/articles.py         tool output → article records → bodies
src/modules/llm.py              the chat model, via OpenRouter
src/modules/mcp_servers/        vendored google-rss-mcp (offline fallback only)
tests/                          contract, connection, and graph tests
```

## Notes

- **The vendored server is a snapshot, not a fork.** It exists so `MCP_MODE=local`
  works with no network. Keep it in sync with upstream — a drifting copy is worse
  than none, because the fallback would answer with a different tool contract than
  the remote server.
- **The session is held open for the whole run.** `MultiServerMCPClient.get_tools`
  opens a fresh session per tool call, which for the stdio modes means a new
  server subprocess each time. Reusing one session cut four `read_article` calls
  from 1.5s to 0.6s locally, and 2.0s to 0.9s over HTTP.
- **A paywall is not an empty result.** Publishers like the NYT return 403 to
  automated requests. That message is kept verbatim on the article so the answer
  can say the body was unavailable rather than implying there was no news.
- **The `stdio` and `local` servers run silenced.** Their banner, request timing,
  and tracebacks go to stderr, and a routine 403 paywall arrives there as a full
  traceback. Run with `LOG_LEVEL=INFO` to get the child's own logs back.
- **The project moved from Poetry to uv**, matching google-rss-mcp.
  `pyproject.toml` is PEP 621 now, `uv.lock` replaces `poetry.lock`, and the dev
  tools live in a `[dependency-groups]` block that `uv sync` installs by default.
  It is declared `package = false`: this is an application run from the repo
  root, not a library to build. `langchain`, `langchain-community`, `requests`,
  `tqdm`, and `mcp` are gone — nothing imports them any more.
