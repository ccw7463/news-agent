"""Tests for choosing and building the connection to google-rss-mcp."""

from contextlib import asynccontextmanager

import pytest

from src.modules import mcp_client


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Start each test from an unconfigured environment."""
    for name in (
        "MCP_MODE",
        "SMITHERY_API_KEY",
        "SMITHERY_URL",
        "MCP_HTTP_URL",
        "GOOGLE_RSS_LANGUAGE",
        "GOOGLE_RSS_REGION",
        "LOG_LEVEL",
        "FASTMCP_LOG_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)


def test_chain_skips_smithery_without_a_key():
    """The gateway 401s anonymously, so trying it keyless only wastes a round trip."""
    assert mcp_client._chain() == ["http", "stdio", "local"]


def test_chain_starts_at_smithery_with_a_key(monkeypatch):
    monkeypatch.setenv("SMITHERY_API_KEY", "sk-test")
    assert mcp_client._chain() == ["smithery", "http", "stdio", "local"]


def test_explicit_mode_disables_fallback(monkeypatch):
    """Pinning a mode must fail loudly rather than quietly serve from the copy."""
    monkeypatch.setenv("MCP_MODE", "smithery")
    assert mcp_client._chain() == ["smithery"]


def test_smithery_without_a_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="SMITHERY_API_KEY"):
        mcp_client._connection("smithery")


def test_smithery_sends_a_bearer_token(monkeypatch):
    monkeypatch.setenv("SMITHERY_API_KEY", "sk-test")
    conn = mcp_client._connection("smithery")
    assert conn["transport"] == "streamable_http"
    assert conn["headers"]["Authorization"] == "Bearer sk-test"


def test_http_needs_no_credentials():
    conn = mcp_client._connection("http")
    assert conn["transport"] == "streamable_http"
    assert "headers" not in conn
    assert conn["url"].endswith("/mcp")


def test_urls_are_overridable(monkeypatch):
    """A moved gateway hostname should not require a code change."""
    monkeypatch.setenv("MCP_HTTP_URL", "https://example.test/mcp")
    assert mcp_client._connection("http")["url"] == "https://example.test/mcp"


def test_local_mode_runs_the_vendored_package_as_a_module():
    """The vendored copy uses relative imports, so a file path would not import."""
    conn = mcp_client._connection("local")
    assert conn["args"] == ["-m", "src.modules.mcp_servers.server"]


def test_locale_is_passed_only_to_servers_we_launch(monkeypatch):
    monkeypatch.setenv("GOOGLE_RSS_LANGUAGE", "ko")
    monkeypatch.setenv("GOOGLE_RSS_REGION", "KR")
    env = mcp_client._connection("local")["env"]
    assert env["GOOGLE_RSS_LANGUAGE"] == "ko"
    assert env["GOOGLE_RSS_REGION"] == "KR"
    assert "env" not in mcp_client._connection("http")


def test_launch_env_keeps_path_so_the_child_can_start(monkeypatch):
    """A bare override dict would launch the server with no PATH and no HOME."""
    monkeypatch.setenv("GOOGLE_RSS_LANGUAGE", "ko")
    env = mcp_client._connection("stdio")["env"]
    assert "PATH" in env
    assert env["GOOGLE_RSS_LANGUAGE"] == "ko"


def test_child_log_level_is_quiet_by_default(monkeypatch):
    """The child's request timing goes to stderr and would clutter the run."""
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert mcp_client._connection("local")["env"]["LOG_LEVEL"] == "CRITICAL"


def test_child_log_level_is_overridable(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    assert mcp_client._connection("local")["env"]["LOG_LEVEL"] == "DEBUG"


def test_child_stderr_is_silenced():
    """A routine 403 paywall otherwise reaches the user as a full traceback."""
    env = mcp_client._connection("local")["env"]
    assert env["FASTMCP_SHOW_SERVER_BANNER"] == "false"
    assert env["FASTMCP_LOG_ENABLED"] == "false"


def test_child_logging_can_be_turned_back_on(monkeypatch):
    """Silencing the child must not make it undebuggable."""
    monkeypatch.setenv("FASTMCP_LOG_ENABLED", "true")
    assert mcp_client._connection("local")["env"]["FASTMCP_LOG_ENABLED"] == "true"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown MCP_MODE"):
        mcp_client._connection("wat")


@pytest.mark.asyncio
async def test_chain_falls_through_to_a_working_mode(monkeypatch):
    """A dead remote must hand off to the next mode, not end the run."""
    attempted = []

    class FakeSession:
        pass

    class FakeClient:
        def __init__(self, config):
            self.mode = next(iter(config.values()))["transport"]

        @asynccontextmanager
        async def session(self, name):
            attempted.append(self.mode)
            if self.mode == "streamable_http":
                raise ConnectionError("gateway down")
            yield FakeSession()

    async def fake_load(session):
        return ["search_news", "get_top_headlines", "read_article"]

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mcp_client, "load_mcp_tools", fake_load)

    async with mcp_client.news_tools() as (tools, mode):
        assert mode == "stdio"
        assert len(tools) == 3


@pytest.mark.asyncio
async def test_every_mode_failing_reports_all_of_them(monkeypatch):
    """One line per mode; a single 'connection failed' hides which one broke."""

    class DeadClient:
        def __init__(self, config):
            pass

        @asynccontextmanager
        async def session(self, name):
            raise ConnectionError("nope")
            yield  # pragma: no cover

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", DeadClient)
    with pytest.raises(RuntimeError) as excinfo:
        async with mcp_client.news_tools():
            pass
    message = str(excinfo.value)
    for mode in ("http", "stdio", "local"):
        assert mode in message


@pytest.mark.asyncio
async def test_the_session_is_closed_when_the_block_ends(monkeypatch):
    """One session for the whole run means one thing to close at the end."""
    closed = []

    class FakeClient:
        def __init__(self, config):
            pass

        @asynccontextmanager
        async def session(self, name):
            try:
                yield object()
            finally:
                closed.append(True)

    monkeypatch.setattr(mcp_client, "MultiServerMCPClient", FakeClient)
    monkeypatch.setattr(mcp_client, "load_mcp_tools", lambda s: _tools())

    async with mcp_client.news_tools() as (tools, _mode):
        assert tools
        assert closed == []
    assert closed == [True]


async def _tools():
    return ["search_news", "read_article"]


def test_the_gateway_url_has_no_mcp_path(monkeypatch):
    """The registry's deploymentUrl is the bare host; /mcp on it answers 404."""
    monkeypatch.setenv("SMITHERY_API_KEY", "sk-test")
    assert not mcp_client._connection("smithery")["url"].endswith("/mcp")
    assert mcp_client._connection("smithery")["url"] == mcp_client.DEFAULT_SMITHERY_URL


def test_the_child_is_pinned_to_stdio():
    """MCP_TRANSPORT=http in the shell would make the child bind a port instead."""
    import os as _os

    _os.environ["MCP_TRANSPORT"] = "http"
    try:
        assert mcp_client._connection("local")["env"]["MCP_TRANSPORT"] == "stdio"
        assert mcp_client._connection("stdio")["env"]["MCP_TRANSPORT"] == "stdio"
    finally:
        _os.environ.pop("MCP_TRANSPORT", None)


def test_exception_groups_are_unwrapped_to_the_real_cause():
    """Otherwise every failure reads 'unhandled errors in a TaskGroup'."""
    inner = ConnectionError("401 Unauthorized")
    grouped = ExceptionGroup("unhandled errors in a TaskGroup", [inner])
    described = mcp_client._describe(grouped)
    assert "401 Unauthorized" in described
    assert "TaskGroup" not in described


def test_nested_exception_groups_are_flattened():
    inner = OSError("nodename nor servname provided")
    described = mcp_client._describe(
        ExceptionGroup("outer", [ExceptionGroup("inner", [inner])])
    )
    assert "nodename nor servname provided" in described


def test_a_plain_exception_is_described_unchanged():
    assert mcp_client._describe(ValueError("boom")) == "ValueError: boom"
