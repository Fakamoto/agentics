# Agentics

Minimalist Python library for LLM usage

* Zero boilerplate API - just `llm("Hello!")`
* Native function calling with tools
* Structured outputs via Pydantic
* Single dependency (openai)

## Installation

```bash
pip install agentics
```

## Why Agentics?

Compare:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

To this:

```python
from agentics import LLM

llm = LLM()
response: str = llm("Hello!")
print(response)
```

## Quickstart

### Simple Chat

```python
from agentics import LLM

llm = LLM(system_prompt="You know everything about the world")

response: str = llm("What is the capital of France?")

print(response)
# The capital of France is Paris.
```

### Structured Output

```python
from agentics import LLM
from pydantic import BaseModel

class ExtractUser(BaseModel):
    name: str
    age: int

llm = LLM()

res = llm.chat("John Doe is 30 years old.", response_format=ExtractUser)

assert res.name == "John Doe"
assert res.age == 30
```

### Tool Usage

```python
from agentics import LLM
import requests

def visit_url(url: str):
    """Fetch the content of a URL"""
    return requests.get(url).content.decode()

llm = LLM()

res = llm.chat("What's the top story on Hacker News?", tools=[visit_url])

print(res)
# The top story on Hacker News is: "Operating System in 1,000 Lines – Intro"
```

### Tool Usage with Structured Output

```python
from agentics import LLM
from pydantic import BaseModel

class HackerNewsStory(BaseModel):
    title: str
    points: int

llm = LLM()

res = llm.chat(
    "What's the top story on Hacker News?", 
    tools=[visit_url], 
    response_format=HackerNewsStory
)

print(res)
# title='Operating System in 1,000 Lines – Intro' points=29
```

### Multiple Tools with Structured Output

```python
from agentics import LLM
from pydantic import BaseModel

def calculate_area(width: float, height: float):
    """Calculate the area of a rectangle"""
    return width * height

def calculate_volume(area: float, depth: float):
    """Calculate volume from area and depth"""
    return area * depth

class BoxDimensions(BaseModel):
    width: float
    height: float
    depth: float
    area: float
    volume: float

llm = LLM()

res = llm.chat(
    "Calculate the area and volume of a box that is 5.5 meters wide, 3.2 meters high and 2.1 meters deep", 
    tools=[calculate_area, calculate_volume],
    response_format=BoxDimensions
)

print(res)
# width=5.5 height=3.2 depth=2.1 area=17.6 volume=36.96
```

# API Reference

## LLM

The main interface for interacting with language models through chat completions. Provides a flexible and minimal API for handling conversations, function calling, and structured outputs.

### Constructor Parameters

- `system_prompt` (str, optional): Initial system prompt to set context. Example:
  ```python
  llm = LLM(system_prompt="You are a helpful assistant")
  ```

- `model` (str, optional): The model identifier to use (default: "gpt-4o-mini")
- `client` (OpenAI, optional): Custom OpenAI client instance. Useful for alternative providers:
  ```python
  client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
  llm = LLM(client=client, model="deepseek-chat")
  ```

- `messages` (list[dict], optional): Pre-populate conversation history:
  ```python
  llm = LLM(messages=[{"role": "user", "content": "Initial message"}])
  ```

### Chat Method

Both `llm.chat()` and `llm()` provide identical functionality as the main interface for interactions.

#### Parameters

- `prompt` (str, optional): The input prompt to send to the model. If provided, appended to conversation history.
- `tools` (list[dict], optional): List of available function tools the model can use. Each tool should be a callable with type hints.
- `response_format` (BaseModel, optional): Pydantic model to structure and validate the response.
- `messages` (list[dict], optional): Override the conversation history for this specific request.
- `single_tool_call_request` (bool, optional): When True, limits the model to one request to use tools (can still call multiple tools in that request).
- `**kwargs`: Additional arguments passed directly to the chat completion API.

#### Return Value
- `Union[str, BaseModel]`: Either a string response or structured data matching response_format

#### Behavior Flows

1. Basic Chat (no tools/response_format):
   - Simple text completion
   - Returns string response

2. With Tools:
   - Model can choose to use available tools or respond directly
   - When tools are used, multiple tools can be called in a single request
   - Tools are called automatically and results fed back
   - Process repeats if model decides to use tools again
   - Use `single_tool_call_request=True` to limit the model to one request to use tools (can still call multiple tools in that request).

3. With Response Format:
   - Response is cast to specified Pydantic model
   - Returns structured data

4. Combined Tools + Response Format:
   - Follows tool flow first
   - Final text response is cast to model

The conversation history is accessible via the `.messages` attribute, making it easy to inspect or manipulate the context.




## Inspiration

Agentics was born from a desire to simplify LLM interactions in Python. The existing landscape often requires verbose boilerplate:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```


When my goal in mind was to be able to simply do `llm("Hello!")`, with that desired interface is how I started building Agentics, this:

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

now turns into this:

```python
from agentics import LLM

llm = LLM()
response = llm("Hello!")
print(response)
```



Agentics makes things simple while bringing these powerful features into the same library:

- **Simple API**: Talk to LLMs with just a few lines of code
- **Structured Output**: Like [instructor](https://github.com/instructor-ai/instructor), turns responses into Pydantic models
- **Function Calling**: Like [Marvin's assistants](https://www.askmarvin.ai/docs/interactive/assistants/) but 
using direct message-based communication instead of the 
Assistants API

I built this to make working with OpenAI's LLMs easier. It handles structured outputs and function calling without any fuss. Right now it only works with OpenAI, but it makes common LLM tasks way simpler.
