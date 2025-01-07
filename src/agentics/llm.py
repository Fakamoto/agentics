from pydantic import BaseModel, Field
from openai import OpenAI
from .utils import (
    create_tool_schema,
    execute_tool,
    format_tool_output,
    system_message,
    user_message,
    assistant_message,
    tool_calls_message,
    tool_message,
)


class LLM:
    """
    A class for interacting with language models through chat completions.

    This class provides a flexible interface for chat-based interactions with language models,
    supporting structured outputs, tool usage, and conversation management.

    Args:
        system_prompt (str, optional): Initial system prompt to set context. Defaults to None.
        model (str, optional): The model identifier to use. Defaults to "gpt-4o-mini".
        client (OpenAI, optional): OpenAI client instance. If None, creates new instance.
        messages (list[dict], optional): Initial conversation messages. Defaults to None.

    Attributes:
        client (OpenAI): The OpenAI client instance
        system_prompt (str): The system prompt used for context
        model (str): The model identifier being used
        messages (list[dict]): The conversation history
    """

    def __init__(
        self,
        system_prompt: str = None,
        model: str = "gpt-4o-mini",
        client: OpenAI = None,
        messages: list[dict] = None,
    ):
        self.client = client or OpenAI()
        self.system_prompt = system_prompt
        self.model = model
        self.messages = messages or []
        if self.system_prompt:
            self.messages.append(system_message(self.system_prompt))

    def __call__(
        self,
        prompt: str = None,
        tools: list[dict] = None,
        response_format: BaseModel = None,
        messages: list[dict] = None,
        single_tool_call_request: bool = False,
        **kwargs,
    ):
        """
        Callable interface to chat method.

        Args:
            prompt (str, optional): The input prompt. Defaults to None.
            tools (list[dict], optional): List of available tools. Defaults to None.
            response_format (BaseModel, optional): Expected response format. Defaults to None.
            messages (list[dict], optional): Override conversation history. Defaults to None.
            single_tool_call_request (bool, optional): Whether to allow only one tool call. Defaults to False.
            **kwargs: Additional arguments passed to chat method.

        Returns:
            Union[str, BaseModel]: The model's response
        """
        return self.chat(prompt, tools, response_format, messages, single_tool_call_request, **kwargs)

    def _chat(self, messages=None, tools=None, **kwargs):
        """
        Internal method for raw chat completions.

        Args:
            messages (list[dict], optional): Messages to send. Defaults to None.
            tools (list[dict], optional): Available tools. Defaults to None.
            **kwargs: Additional arguments passed to chat completion.

        Returns:
            ChatCompletion: Raw completion response from the model
        """
        params = {"model": self.model, "messages": messages or self.messages, **kwargs}
        if tools:
            params["tools"] = tools

        completion = self.client.chat.completions.create(**params)
        return completion

    def _cast(self, response_format=None, messages=None, tools=None, **kwargs):
        """
        Internal method for structured chat completions.

        Args:
            response_format (BaseModel, optional): Expected response format. Defaults to None.
            messages (list[dict], optional): Messages to send. Defaults to None.
            tools (list[dict], optional): Available tools. Defaults to None.
            **kwargs: Additional arguments passed to chat completion.

        Returns:
            ChatCompletion: Parsed completion response with structured data
        """
        params = {
            "model": self.model,
            "messages": messages or self.messages,
            "response_format": response_format,
            **kwargs,
        }
        if tools:
            params["tools"] = tools

        completion = self.client.beta.chat.completions.parse(**params)
        return completion

    def cast(self, prompt: str, response_format=None):
        """
        Single structured chat completion without saving to conversation.

        Args:
            prompt (str): The input prompt
            response_format (BaseModel, optional): Expected response format. Defaults to None.

        Returns:
            BaseModel: Structured response matching response_format schema
        """
        messages = [user_message(prompt)]
        completion = self._cast(messages=messages, response_format=response_format)
        return completion.choices[0].message.parsed

    def chat(
        self,
        prompt: str = None,
        tools: list[dict] = None,
        response_format: BaseModel = None,
        messages: list[dict] = None,
        single_tool_call_request: bool = False,
        **kwargs,
    ):
        """
        Main method for chat completions with full functionality.

        Supports structured outputs, tool usage, and maintains conversation history.
        Can handle both regular text responses and tool-based interactions.

        Args:
            prompt (str, optional): The input prompt. Defaults to None.
            tools (list[dict], optional): Available tools. Defaults to None.
            response_format (BaseModel, optional): Expected response format. Defaults to None.
            messages (list[dict], optional): Override conversation history. Defaults to None.
            single_tool_call_request (bool, optional): Whether to allow only one tool call. Defaults to False.
            **kwargs: Additional arguments passed to chat completion.

        Returns:
            Union[str, BaseModel]: The model's response, either as text or structured data.
                If response_format is provided, returns validated BaseModel instance.
                If no response_format, returns string response.
                For tool calls, returns the final response after tool execution.

        Raises:
            ValueError: If no response is received from the model
        """
        if prompt:
            self.messages.append(user_message(prompt))

        if tools:
            tools = [
                create_tool_schema(tool, strict=True if response_format else False)
                for tool in tools
            ]

        if response_format:
            completion = self._cast(
                response_format=response_format,
                tools=tools,
                messages=messages or self.messages,
                **kwargs,
            )
        else:
            completion = self._chat(
                messages=messages or self.messages, tools=tools, **kwargs
            )

        choice = completion.choices[0]

        if choice.finish_reason != "tool_calls":
            if response_format and choice.message.parsed:
                validated_data: BaseModel = choice.message.parsed
                raw_response = choice.message.content
                self.messages.append(assistant_message(raw_response))
                return validated_data

            elif choice.message.content:
                text_response = choice.message.content
                self.messages.append(assistant_message(text_response))
                return text_response
            else:
                raise ValueError("No response from the model")

        elif choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls
            self.messages.append(tool_calls_message(tool_calls))

            for tool_call in tool_calls:
                output = execute_tool(
                    tools=tools,
                    function_name=tool_call.function.name,
                    function_arguments_json=tool_call.function.arguments,
                )
                string_output = format_tool_output(output)
                tool_output_message = tool_message(
                    name=tool_call.function.name,
                    tool_call_id=tool_call.id,
                    content=string_output,
                )
                self.messages.append(tool_output_message)

            params = {
                "response_format": response_format,
                "tools": tools,
                "messages": messages or self.messages,
                **kwargs,
            }

            if single_tool_call_request:
                params["tools"] = None

            response: str = self.chat(**params)

            return response
