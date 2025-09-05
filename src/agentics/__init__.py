import importlib.metadata

from .llm import LLM
from .embedding import Embedding
from .utils import (
    system_message,
    user_message,
    assistant_message,
    tool_message,
    tool_calls_message,
)
from .dspy_core import Program

__version__ = importlib.metadata.version("agentics")

__all__ = [
    # Main classes
    "LLM",
    "Embedding",
    # Utility functions
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
    "tool_calls_message",

    # Dspy Core
    "Program",
]
