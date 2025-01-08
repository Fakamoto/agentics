# Agentics

A minimal LLM agent library that provides a simple interface for chat-based interactions with language models, supporting structured outputs and tool usage.

## Installation

```bash
pip install agentics
```

## Usage

### Simple Chat

```python
from agentics import LLM

llm = LLM()
response = llm("What is the capital of France?")
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
response: ExtractUser = llm.chat("John Doe is 30 years old.", response_format=ExtractUser)

assert response.name == "John Doe"
assert response.age == 30
```

### Tool Usage

```python
from agentics import LLM
import requests

def visit_url(url: str):
    """Fetch the content of a URL"""
    return requests.get(url).content.decode()

llm = LLM()
response = llm.chat("What's the top story on Hacker News?", tools=[visit_url])
print(response)
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
response: HackerNewsStory = llm.chat(
    "What's the top story on Hacker News?", 
    tools=[visit_url], 
    response_format=HackerNewsStory
)
print(response)
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

response: BoxDimensions = llm.chat(
    "Calculate the area and volume of a box that is 5.5 meters wide, 3.2 meters high and 2.1 meters deep", 
    tools=[calculate_area, calculate_volume],
    response_format=BoxDimensions
)

print(response)
# width=5.5 height=3.2 depth=2.1 area=17.6 volume=36.96
```

### Weather Information with Multiple Tools

```python
def get_temperature(city: str):
    """Get the current temperature for a city"""
    # Simulated API response
    return 22.5

def convert_to_fahrenheit(celsius: float):
    """Convert Celsius to Fahrenheit"""
    return (celsius * 9/5) + 32

class WeatherInfo(BaseModel):
    city: str
    celsius: float
    fahrenheit: float

llm = LLM()

response: WeatherInfo = llm.chat(
    "What's the temperature in Paris? Convert it to Fahrenheit.",
    tools=[get_temperature, convert_to_fahrenheit],
    response_format=WeatherInfo
)

print(response)
# city='Paris' celsius=22.5 fahrenheit=72.5
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
