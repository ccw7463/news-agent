"""The chat model, served through OpenRouter.

OpenRouter speaks the OpenAI wire format, so the OpenAI client works unchanged
once it is pointed at OpenRouter's base URL. That is the whole integration —
switching models is an environment variable, not a code change.
"""

import os

from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

# Summaries should not drift between runs over the same article, and tool-call
# arguments should not be creative at all.
DEFAULT_TEMPERATURE = 0.2


def build_model() -> ChatOpenAI:
    """Build the chat model from the environment.

    Returns:
        A model pointed at OpenRouter, ready for ``bind_tools``.

    Raises:
        RuntimeError: If no API key is configured.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill "
            "it in; get a key at https://openrouter.ai/keys."
        )

    temperature = os.environ.get("LLM_TEMPERATURE", "").strip()
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL,
        base_url=os.environ.get("OPENROUTER_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        api_key=key,
        temperature=float(temperature) if temperature else DEFAULT_TEMPERATURE,
        # OpenRouter attributes usage to these; they are optional and carry no
        # credentials.
        default_headers={
            "HTTP-Referer": "https://github.com/ccw7463/news-agent",
            "X-Title": "news-agent",
        },
    )


def model_label() -> str:
    """Name the configured model, for display.

    Returns:
        The model id that :func:`build_model` will use.
    """
    return os.environ.get("OPENROUTER_MODEL", "").strip() or DEFAULT_MODEL
