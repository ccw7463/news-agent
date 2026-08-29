"""Vendored snapshot of the google-rss-mcp server.

This is a copy of https://github.com/ccw7463/google-rss-mcp, kept so the agent
still works with no network path to the hosted instance. It is a *fallback*:
the normal path is ``MCP_MODE=smithery`` or ``http`` in ``src/modules/mcp_client.py``.

Keep this in sync with upstream. A drifting copy is worse than none, because the
fallback would answer with a different tool contract than the remote server.
"""

__version__ = "0.2.0"
__upstream__ = "https://github.com/ccw7463/google-rss-mcp"

from .config import Settings
from .rss import Article, GoogleNewsClient, GoogleNewsError, NewsItem

__all__ = [
    "__version__",
    "Article",
    "GoogleNewsClient",
    "GoogleNewsError",
    "NewsItem",
    "Settings",
]
