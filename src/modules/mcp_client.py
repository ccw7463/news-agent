"""Connection to the google-rss-mcp server, with a fallback chain.

The same server is reachable four ways, in descending order of "this is the
published thing":

``smithery``
    The Smithery gateway, which proxies the registry listing at
    ``@ccw7463/google-rss-mcp``. Needs ``SMITHERY_API_KEY``: the gateway is an
    OAuth 2.0 protected resource and answers 401 to every anonymous request,
    on every one of its hostnames. That is Smithery's access-control layer in
    front of the server, not anything the server itself requires.
``http``
    The origin instance the gateway proxies to. No key, no account.
``stdio``
    Upstream source pulled and run locally by ``uvx``. Needs network to GitHub
    once, then runs offline. Lets you pin a locale for this process only.
``local``
    The vendored snapshot in ``src/modules/mcp_servers``. Needs no network at
    all, and is the last resort.

Pick one with ``MCP_MODE``. Left unset, the chain starts at ``smithery`` when a
key is present and at ``http`` otherwise, then walks down the list on failure.
Every mode speaks the same three tools with the same schemas, so the graph
downstream does not care which one answered.
"""

import logging
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

SERVER_NAME = "google-rss-mcp"


def _describe(exc: BaseException) -> str:
    """Render an exception in a form that says what actually went wrong.

    Both MCP transports run inside anyio task groups, so a 401, a DNS failure,
    and a 500 all arrive as ``ExceptionGroup`` — whose ``str()`` is only a count
    of sub-exceptions. Reporting that verbatim makes every connection failure
    look identical.

    Args:
        exc: The exception to describe.

    Returns:
        A one-line description naming the underlying causes.
    """
    group = getattr(exc, "exceptions", None)
    if group:
        inner = ", ".join(_describe(e) for e in group)
        return inner or f"{type(exc).__name__}: {exc}"
    return f"{type(exc).__name__}: {exc}"


# The gateway for the published listing, verbatim from the registry's
# `deploymentUrl`. Note there is no `/mcp` path: the root is the endpoint, and
# `/mcp` on this host answers 404. Overridable because Smithery has changed this
# hostname form before, and a stale constant here would look like an outage.
#   curl https://registry.smithery.ai/servers/@ccw7463/google-rss-mcp
DEFAULT_SMITHERY_URL = "https://google-rss-mcp--ccw7463.run.tools"
DEFAULT_HTTP_URL = "https://google-rss-mcp-production.up.railway.app/mcp"
UPSTREAM_REPO = "git+https://github.com/ccw7463/google-rss-mcp"

MODES = ("smithery", "http", "stdio", "local")


def _launch_env() -> Dict[str, str]:
    """Build the environment for a server we start ourselves.

    Locale is only meaningful here: a shared remote instance stays neutral and
    takes language and region per tool call instead.

    Built by merging onto ``os.environ`` rather than from scratch. The MCP SDK
    takes whatever dict it is given literally, so a bare
    ``{"GOOGLE_RSS_LANGUAGE": "ko"}`` would launch the child with no ``PATH``
    and no ``HOME`` — and ``uvx`` would not even be found.

    The child is also silenced. Its banner, request timing, and tracebacks all
    go to stderr and would interleave with the agent's own output — and a 403
    paywall, which is routine, arrives there as a full traceback. Nothing is
    lost: a failed tool call still comes back in the tool result, where the
    agent reports it, and a server that will not start still surfaces as a
    connection failure naming the mode. Export ``LOG_LEVEL=INFO`` to get the
    child's own logs back when that is not enough.

    Returns:
        The full child environment.
    """
    env = {
        **os.environ,
        # Inherited from os.environ, this would be whatever the operator set for
        # a deployment. The server reads it, so MCP_TRANSPORT=http would make the
        # child bind a port and never speak stdio — the handshake then hangs
        # until timeout instead of failing. We are launching it on a pipe.
        "MCP_TRANSPORT": "stdio",
        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "CRITICAL"),
        "FASTMCP_SHOW_SERVER_BANNER": os.environ.get(
            "FASTMCP_SHOW_SERVER_BANNER", "false"
        ),
        "FASTMCP_LOG_ENABLED": os.environ.get("FASTMCP_LOG_ENABLED", "false"),
    }
    for name in ("GOOGLE_RSS_LANGUAGE", "GOOGLE_RSS_REGION"):
        value = os.environ.get(name, "").strip()
        if value:
            env[name] = value
    return env


def _connection(mode: str) -> Dict[str, Any]:
    """Build the langchain-mcp-adapters connection dict for one mode.

    Args:
        mode: One of ``MODES``.

    Returns:
        A connection dict ready for ``MultiServerMCPClient``.

    Raises:
        RuntimeError: If ``mode`` is ``smithery`` and no API key is set.
        ValueError: If ``mode`` is not a known mode.
    """
    if mode == "smithery":
        key = os.environ.get("SMITHERY_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "SMITHERY_API_KEY is not set. The Smithery gateway rejects "
                "anonymous requests with 401; get a key at "
                "https://smithery.ai/account/api-keys, or use MCP_MODE=http."
            )
        return {
            "transport": "streamable_http",
            "url": os.environ.get("SMITHERY_URL", "").strip() or DEFAULT_SMITHERY_URL,
            "headers": {"Authorization": f"Bearer {key}"},
        }

    if mode == "http":
        return {
            "transport": "streamable_http",
            "url": os.environ.get("MCP_HTTP_URL", "").strip() or DEFAULT_HTTP_URL,
        }

    if mode == "stdio":
        conn = {
            "transport": "stdio",
            "command": "uvx",
            "args": ["--from", UPSTREAM_REPO, "google-rss-mcp"],
        }
        conn["env"] = _launch_env()
        return conn

    if mode == "local":
        conn = {
            "transport": "stdio",
            # -m, not a file path: the vendored package uses relative imports,
            # so it has to be imported as a package rather than run as a script.
            "command": sys.executable,
            "args": ["-m", "src.modules.mcp_servers.server"],
        }
        conn["env"] = _launch_env()
        return conn

    raise ValueError(f"unknown MCP_MODE {mode!r}; expected one of {', '.join(MODES)}")


def _chain() -> List[str]:
    """Decide which modes to try, in order.

    An explicit ``MCP_MODE`` is taken literally — one mode, no fallback — so a
    deployment that must prove it is talking to Smithery fails loudly instead of
    quietly serving from the vendored copy. With nothing set, we walk the full
    chain from wherever credentials allow us to start.

    Returns:
        Mode names to attempt in order.
    """
    requested = os.environ.get("MCP_MODE", "").strip().lower()
    if requested:
        return [requested]
    start = 0 if os.environ.get("SMITHERY_API_KEY", "").strip() else 1
    return list(MODES[start:])


@asynccontextmanager
async def news_tools(
    on_attempt: Optional[Any] = None,
) -> AsyncIterator[Tuple[List[Any], str]]:
    """Connect to the news server and yield its tools over one live session.

    The tools are bound to a single session held open for the whole block.
    ``MultiServerMCPClient.get_tools`` would instead open a fresh session per
    tool call — a new MCP handshake every time, and for the stdio modes a whole
    new server subprocess. A run that reads five articles pays that five times.

    Args:
        on_attempt: Optional ``callable(mode, status, detail)`` for progress
            reporting, where ``status`` is ``"trying"``, ``"ok"``, or ``"failed"``.

    Yields:
        A ``(tools, mode)`` pair naming the mode that answered.

    Raises:
        RuntimeError: If every mode in the chain failed.
    """
    failures = []

    for mode in _chain():
        if on_attempt:
            on_attempt(mode, "trying", "")

        stack = AsyncExitStack()
        try:
            client = MultiServerMCPClient({SERVER_NAME: _connection(mode)})
            session = await stack.enter_async_context(client.session(SERVER_NAME))
            tools = await load_mcp_tools(session)
        except Exception as exc:  # noqa: BLE001 - any failure means "try the next mode"
            await stack.aclose()
            detail = _describe(exc)
            logger.warning("MCP mode %s failed — %s", mode, detail)
            failures.append((mode, detail))
            if on_attempt:
                on_attempt(mode, "failed", detail)
            continue

        if not tools:
            await stack.aclose()
            failures.append((mode, "server returned no tools"))
            if on_attempt:
                on_attempt(mode, "failed", "server returned no tools")
            continue

        if on_attempt:
            on_attempt(mode, "ok", f"{len(tools)} tools")
        try:
            yield tools, mode
        finally:
            await stack.aclose()
        return

    report = "\n".join(f"  - {mode}: {detail}" for mode, detail in failures)
    raise RuntimeError(f"could not reach {SERVER_NAME} in any mode:\n{report}")
