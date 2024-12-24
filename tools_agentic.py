from pydantic import BaseModel, Field
from openai import OpenAI
from utils import create_tool_schema, execute_tool, format_tool_output, system_message, user_message, assistant_message, tool_calls_message, tool_message


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
    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.system_prompt = system_prompt
        self.model = model
        self.messages = []
        if self.system_prompt:
            self.messages.append(system_message(self.system_prompt))

    def chat(self, prompt: str = None, **kwargs) -> str:
        """Chat completion with raw text response"""
        if prompt:
            self.messages.append(user_message(prompt))

        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            **kwargs
        )
        text_response = completion.choices[0].message.content
        self.messages.append(assistant_message(text_response))
        return text_response
    
    def chat_with_tools(self, prompt: str, tools: list[dict], **kwargs):
        """Chat completion with tools"""
        self.messages.append(user_message(prompt))

        tools=[create_tool_schema(tool) for tool in tools]

        completion = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            **kwargs
        )
        choice = completion.choices[0]

        if choice.message.content and choice.finish_reason != "tool_calls":
            text_response = choice.message.content
            self.messages.append(assistant_message(text_response))
            return text_response
        

        if choice.finish_reason == "tool_calls":
            tool_calls = choice.message.tool_calls
            self.messages.append(tool_calls_message(tool_calls))

            for tool_call in tool_calls:
                output = execute_tool(
                    tools=tools,
                    function_name=tool_call.function.name,
                    function_arguments_json=tool_call.function.arguments,
                )
                string_output = format_tool_output(output)
                tool_output_message = tool_message(name=tool_call.function.name, tool_call_id=tool_call.id, content=string_output)
                self.messages.append(tool_output_message)

            response: str = self.chat()
            return response
        

        

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

llm = LLM("you are a helpful assistant that relies on tools to answer questions")

def answer_to_enigma() -> str:
    return "answer is 'mike es papanatas'"

def add(a: int, b: int) -> int:
    return a + b

response = llm.chat_with_tools("what is the answer to the enigma? and what is 2 + 2? use both tools", tools=[answer_to_enigma, add])

print(response)
# The answer to the enigma is "mike es papanatas," and 2 + 2 equals 4.