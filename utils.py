from typing import Any, Callable, TypeVar, Optional, Generic
from pydantic import BaseModel, TypeAdapter, ConfigDict, PrivateAttr
import inspect
import json
import asyncio

T = TypeVar("T")

class ToolFunction(BaseModel, Generic[T]):
    """Represents a callable function with its metadata"""
    name: str
    description: Optional[str] = None
    parameters: dict
    _python_fn: Callable = PrivateAttr()

    @classmethod
    def create(cls, name: str, description: Optional[str], parameters: dict, _python_fn: Callable):
        instance = cls(
            name=name,
            description=description,
            parameters=parameters,
        )
        instance._python_fn = _python_fn
        return instance

class Tool(BaseModel, Generic[T]):
    """OpenAI-compatible function tool"""
    type: str = "function"
    function: ToolFunction[T]

    @classmethod
    def create(cls, function: ToolFunction[T]):
        return cls(type="function", function=function)

def create_tool_schema(
    fn: Callable[..., T],
    name: Optional[str] = None,
    description: Optional[str] = None,
    kwargs: Optional[dict[str, Any]] = None,
) -> Tool[T]:
    """Creates an OpenAI-compatible tool from a Python function."""
    # If kwargs provided, create a simple wrapper function
    if kwargs:
        original_fn = fn
        fn = lambda **args: original_fn(**{**kwargs, **args})
        fn.__name__ = original_fn.__name__
        fn.__doc__ = original_fn.__doc__

    schema = TypeAdapter(
        fn, 
        config=ConfigDict(arbitrary_types_allowed=True)
    ).json_schema()

    return Tool[T].create(
        ToolFunction[T].create(
            name=name or fn.__name__,
            description=description or fn.__doc__,
            parameters=schema,
            _python_fn=fn,
        )
    )

def execute_tool(
    tools: list[Tool[Any]],
    function_name: str,
    function_arguments_json: str,
) -> Any:
    """Helper function for calling a function tool from a list of tools."""
    tool = next(
        (t for t in tools
         if t.function and t.function.name == function_name),
        None
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
