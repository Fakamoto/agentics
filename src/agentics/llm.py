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


client = OpenAI()


class AgentDefinition(BaseModel):
    """
    Minimal representation of an agent's essential information.
    """

    name: str
    description: str
    instructions: str
    objective: str
    motivation: str


class LLM:
    def __init__(
        self,
        system_prompt: str = None,
        model: str = "gpt-4o-mini",
        messages: list[dict] = None,
    ):
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
        **kwargs,
    ):
        return self.chat(prompt, tools, response_format, messages, **kwargs)

    def _chat(self, messages=None, tools=None, **kwargs):
        params = {"model": self.model, "messages": messages or self.messages, **kwargs}
        if tools:
            params["tools"] = tools

        completion = client.chat.completions.create(**params)
        return completion

    def _cast(self, response_format=None, messages=None, tools=None, **kwargs):
        """Chat completion with structured output"""

        params = {
            "model": self.model,
            "messages": messages or self.messages,
            "response_format": response_format,
            **kwargs,
        }
        if tools:
            params["tools"] = tools

        completion = client.beta.chat.completions.parse(**params)
        return completion

    def cast(self, prompt: str, response_format=None):
        """Chat completion with structured output without saving the conversation"""
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
        """Chat completion with tools"""
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
