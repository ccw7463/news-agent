"""Runtime configuration, resolved from environment variables.

Every value has a neutral default so the server works with no configuration at
all. Operators pin their own locale by exporting the variables below; callers
can still override language/region per tool call.
"""

import os
from dataclasses import dataclass

# Neutral defaults so an unconfigured install is useful to the widest audience.
DEFAULT_LANGUAGE = "en"
DEFAULT_REGION = "US"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_MAX_LENGTH = 5000


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _env_number(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class Settings:
    """Server-wide defaults.

    Attributes:
        language: Google News ``hl`` code, e.g. ``en``, ``ko``, ``ja``.
        region: Google News ``gl`` code, e.g. ``US``, ``KR``, ``JP``.
        timeout: Per-request timeout in seconds.
        max_concurrency: Maximum simultaneous outbound HTTP requests.
        max_length: Default article content truncation length in characters.
    """

    language: str = DEFAULT_LANGUAGE
    region: str = DEFAULT_REGION
    timeout: float = DEFAULT_TIMEOUT
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    max_length: int = DEFAULT_MAX_LENGTH

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from ``GOOGLE_RSS_*`` environment variables."""
        return cls(
            language=_env_str("GOOGLE_RSS_LANGUAGE", DEFAULT_LANGUAGE),
            region=_env_str("GOOGLE_RSS_REGION", DEFAULT_REGION),
            timeout=_env_number("GOOGLE_RSS_TIMEOUT", DEFAULT_TIMEOUT, 1, 120),
            max_concurrency=int(
                _env_number(
                    "GOOGLE_RSS_MAX_CONCURRENCY", DEFAULT_MAX_CONCURRENCY, 1, 32
                )
            ),
            max_length=int(
                _env_number("GOOGLE_RSS_MAX_LENGTH", DEFAULT_MAX_LENGTH, 200, 100_000)
            ),
        )
