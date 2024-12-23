from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from utils import create_tool_schema, execute_tool, format_tool_output


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


def system(text: str):
    return {"role": "system", "content": text}

def user(text: str):
    return {"role": "user", "content": text}

def assistant(text: str):
    return {"role": "assistant", "content": text}

def tool_calls(calls: list[ChatCompletionMessageToolCall]) -> dict:
    """Convert tool calls to the proper message format"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments
            }
        } for call in calls]
    }

def tool(output: dict) -> dict:
    """Convert a single tool output to the proper message format."""
    return {
        "role": "tool",
        "tool_call_id": output["tool_call_id"],
        "name": output["name"],
        "content": output["content"]
    }


class LLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        """Chat completion with raw text response"""
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content
    
    def chat_with_tools(self, messages: list[dict], tools: list[dict], **kwargs):
        """Chat completion with tools"""
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            **kwargs
        )
        choice = completion.choices[0]
        if choice.finish_reason == "tool_calls":
            return choice.message.tool_calls
        return choice.message.content

    def cast(self, messages: list[dict], response_format = None):
        """Chat completion with structured output"""
        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
        )
        validated_data = completion.choices[0].message.parsed

        return validated_data






# agent = Agent("Mastermind", "You are the Mastermind, you are tasked with writing a complete book of 100 pages about fire and save it in ./fire.pdf")

llm = LLM()

messages = [user("What is the answer to the enigma? and what is 2 + 2? use both tools")]

def answer_to_enigma() -> str:
    return "answer is 'facu'"

def add(a: int, b: int) -> int:
    return a + b

tools = [answer_to_enigma, add]

tools=[create_tool_schema(tool) for tool in tools]

res = llm.chat_with_tools(messages, tools=tools)


messages.append(tool_calls(res))

tool_outputs = []
for tool_call in res:
    output = execute_tool(
        tools=tools,
        function_name=tool_call.function.name,
        function_arguments_json=tool_call.function.arguments,
    )
    string_output = format_tool_output(output)
    tool_outputs.append(
        dict(
            name=tool_call.function.name,
            tool_call_id=tool_call.id,
            content=string_output,
        )
    )


for output in tool_outputs:
    messages.append(tool(output))

print(messages)
response = llm.chat(messages)


print(response)