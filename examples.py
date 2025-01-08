"""Simple chat"""
from agentics import LLM

llm = LLM()

response: str = llm("What is the capital of France?")

print(response)
# The capital of France is Paris.

"""Structured output"""
from agentics import LLM
from pydantic import BaseModel


class ExtractUser(BaseModel):
    name: str
    age: int


llm = LLM()

response: ExtractUser = llm.chat("John Doe is 30 years old.", response_format=ExtractUser)

assert response.name == "John Doe"
assert response.age == 30


"""Tool usage"""
from agentics import LLM
import requests


# Define a custom tool function as python function
def visit_url(url: str):
    """Fetch the content of a URL"""
    return requests.get(url).content.decode()


llm = LLM()

response: str = llm.chat("What's the top story on Hacker News?", tools=[visit_url])

print(response)
# The top story on Hacker News is: **"Operating System in 1,000 Lines – Intro"**.

"""Tool usage with structured output"""
from agentics import LLM
from pydantic import BaseModel


class HackerNewsStory(BaseModel):
    title: str
    points: int


llm = LLM()

response: HackerNewsStory = llm.chat("What's the top story on Hacker News?", tools=[visit_url], response_format=HackerNewsStory)

print(response)
# title='Operating System in 1,000 Lines – Intro' points=29


"""Multiple tools with structured output"""

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

"""another example of multiple tools with structured output"""

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