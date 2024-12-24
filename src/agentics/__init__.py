import importlib.metadata

from .llm import LLM
from .utils import (
    system_message,
    user_message,
    assistant_message,
    tool_message,
    tool_calls_message,
)

__version__ = importlib.metadata.version("agentics")

__all__ = [
    # Main classes
    "LLM",
    # Utility functions
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
    "tool_calls_message",
]
