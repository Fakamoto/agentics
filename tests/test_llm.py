import pytest
from pydantic import BaseModel
from agentics import LLM


def test_llm_chat():
    """1) Test LLM chatting normally, no tools."""
    llm = LLM()
    response = llm.chat("Hello, how are you?")
    assert isinstance(response, str)
    assert len(response) > 0


def test_llm_chat_with_tool():
    """2) Test LLM chat with one tool, verify a keyword is present in the response."""

    def secret_keyword_tool() -> str:
        return "SECRET_KEYWORD"

    llm = LLM()
    response = llm.chat(
        "Can you reveal the secret keyword?", tools=[secret_keyword_tool]
    )
    assert isinstance(response, str)
    assert "SECRET_KEYWORD" in response


def test_llm_chat_structured_output():
    """3) Check only response format with no tool."""

    class Response(BaseModel):
        message: str

    llm = LLM()
    result = llm.chat("say hi.", response_format=Response)
    assert isinstance(result, Response)
    assert isinstance(result.message, str)


def test_llm_chat_multiple_tools():
    """4) Check multiple tool usage, ensure multiple keywords are present in final response."""

    def first_secret_tool() -> str:
        return "FIRST_SECRET"

    def second_secret_tool() -> str:
        return "SECOND_SECRET"

    llm = LLM()
    response = llm.chat(
        "Please reveal two secrets together.",
        tools=[first_secret_tool, second_secret_tool],
    )
    assert isinstance(response, str)
    assert "FIRST_SECRET" in response
    assert "SECOND_SECRET" in response


def test_llm_chat_tool_with_structured_output():
    """
    5) Check tool usage with structured output,
    ensure field is int and has the correct value.
    """

    class IntResponse(BaseModel):
        number: int

    def secret_number() -> int:
        return 55

    llm = LLM()
    result = llm.chat(
        "what is the secret number?", tools=[secret_number], response_format=IntResponse
    )
    assert isinstance(result, IntResponse)
    assert result.number == 55


def test_llm_chat_multiple_tools_with_structured_output():
    """
    6) Like the previous test but with multiple tools
    and structured output with multiple fields.
    """

    class MultiToolResponse(BaseModel):
        text: str
        first_number: int
        second_number: int

    def first_secret_number() -> int:
        return 10

    def second_secret_number() -> int:
        return 20

    llm = LLM()
    result = llm.chat(
        "What are the secret numbers?",
        tools=[first_secret_number, second_secret_number],
        response_format=MultiToolResponse,
    )
    assert isinstance(result, MultiToolResponse)
    assert result.first_number == 10
    assert result.second_number == 20


# Additional Functional Tests
def test_llm_empty_tools_list():
    """Test behavior with empty tools list."""
    llm = LLM()
    response = llm.chat("Hello", tools=[])
    assert isinstance(response, str)


def test_llm_complex_structured_output():
    """Test nested structured output."""

    class NestedModel(BaseModel):
        value: str

    class ComplexResponse(BaseModel):
        field1: str
        nested: NestedModel

    llm = LLM()
    result = llm.chat("Return nested structure", response_format=ComplexResponse)
    assert isinstance(result, ComplexResponse)
    assert isinstance(result.nested, NestedModel)


# Quality/Behavior Tests
def test_llm_math_accuracy():
    """Test mathematical accuracy using tools."""

    def multiply(a: int, b: int) -> int:
        return a * b

    class MathResult(BaseModel):
        result: int

    llm = LLM()
    response = llm.chat(
        "What is 6 times 7? Use the multiply tool.",
        tools=[multiply],
        response_format=MathResult,
    )
    assert response.result == 42


def test_llm_tool_choice():
    """Test if model chooses correct tool when given options."""

    def get_weather() -> str:
        return "sunny"

    def get_time() -> str:
        return "12:00"

    llm = LLM()
    response = llm.chat("What's the weather like?", tools=[get_weather, get_time])
    assert "sunny" in response.lower()


def test_llm_call_operator():
    """Test the __call__ operator with tools and structured output."""

    def add(a: int, b: int) -> int:
        return a + b

    class AddResult(BaseModel):
        sum: int

    llm = LLM()
    response = llm(
        "What is 5 plus 3? Use the add tool.", tools=[add], response_format=AddResult
    )
    assert isinstance(response, AddResult)
    assert response.sum == 8
