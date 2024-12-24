from typing import Any, Callable, TypeVar, Optional, Generic
from pydantic import BaseModel, TypeAdapter, ConfigDict, PrivateAttr
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)
import inspect
import json
import asyncio

T = TypeVar("T")


def system_message(text: str):
    return {"role": "system", "content": text}


def user_message(text: str):
    return {"role": "user", "content": text}


def assistant_message(text: str):
    return {"role": "assistant", "content": text}


def tool_calls_message(calls: list[ChatCompletionMessageToolCall]) -> dict:
    """Convert tool calls to the proper message format"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in calls
        ],
    }


def tool_message(name: str, tool_call_id: str, content: str) -> dict:
    """Convert a single tool output to the proper message format."""
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": content,
    }


class ToolFunction(BaseModel, Generic[T]):
    """Represents a callable function with its metadata"""

    name: str
    description: Optional[str] = None
    parameters: dict
    strict: bool = False
    _python_fn: Callable = PrivateAttr()

    @classmethod
    def create(
        cls,
        name: str,
        description: Optional[str],
        parameters: dict,
        _python_fn: Callable,
        strict: bool = False,
    ):
        instance = cls(
            name=name, description=description, parameters=parameters, strict=strict
        )
        instance._python_fn = _python_fn
        return instance

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


class Tool(BaseModel, Generic[T]):
    """OpenAI-compatible function tool"""

    type: str = "function"
    function: ToolFunction[T]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def create(cls, function: ToolFunction[T]):
        return cls(type="function", function=function)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def create_tool_schema(
    fn: Callable[..., T] | Tool[T],
    name: Optional[str] = None,
    description: Optional[str] = None,
    kwargs: Optional[dict[str, Any]] = None,
    strict: bool = False,
) -> Tool[T]:
    """Creates an OpenAI-compatible tool from a Python function or returns existing Tool."""
    if isinstance(fn, Tool):
        return fn

    # If kwargs provided, create a simple wrapper function
    if kwargs:
        original_fn = fn
        fn = lambda **args: original_fn(**{**kwargs, **args})
        fn.__name__ = original_fn.__name__
        fn.__doc__ = original_fn.__doc__

    schema = TypeAdapter(
        fn, config=ConfigDict(arbitrary_types_allowed=True)
    ).json_schema()

    return Tool[T].create(
        ToolFunction[T].create(
            name=name or fn.__name__,
            description=description or fn.__doc__,
            parameters=schema,
            _python_fn=fn,
            strict=strict,
        )
    )


def execute_tool(
    tools: list[Tool[Any]],
    function_name: str,
    function_arguments_json: str,
) -> Any:
    """Helper function for calling a function tool from a list of tools."""
    tool = next(
        (t for t in tools if t.function and t.function.name == function_name), None
    )

    if not tool or not tool.function or not tool.function._python_fn:
        raise ValueError(f"Tool not found: {function_name}")

    arguments = json.loads(function_arguments_json)
    output = tool.function._python_fn(**arguments)

    # Simple async handling
    if inspect.iscoroutine(output):
        output = asyncio.run(output)

    return output


def format_tool_output(output: Any) -> str:
    """Function outputs must be provided as strings"""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    try:
        return TypeAdapter(type(output)).dump_json(output).decode()
    except Exception:
        return str(output)
