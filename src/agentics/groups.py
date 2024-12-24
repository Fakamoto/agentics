import inspect
from typing import Any, List, Optional, Tuple, Type, TypeVar, Union
from pydantic import BaseModel, Field
from openai import OpenAI

T = TypeVar("T", bound=BaseModel)
client = OpenAI()


def system(text: str):
    return {"role": "system", "content": text}


def user(text: str):
    return {"role": "user", "content": text}


def assistant(text: str):
    return {"role": "assistant", "content": text}


class LLM:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.messages: List[dict] = []

    def _chat(self, messages: List[dict] = []) -> str:
        """Chat completion with raw text response"""
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def chat(
        self,
        messages: List[dict] = [],
        response_format: Type[Union[T, str, int, float, bool]] = None,
    ) -> Union[T, str, int, float, bool]:
        """Chat completion with raw text response"""
        if response_format:
            return self._cast(messages, response_format)
        else:
            return self._chat(messages)

    def _cast(
        self,
        messages: List[dict] = [],
        response_format: Type[Union[T, str, int, float, bool]] = None,
    ) -> Any:
        """Chat completion with structured output"""
        is_primitive = response_format in (str, int, float, bool)
        if is_primitive:

            class PrimitiveWrapper(BaseModel):
                value: Any = Field(...)

            response_format = PrimitiveWrapper

        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=messages,
            response_format=response_format,
        )

        validated_data = completion.choices[0].message.parsed

        if is_primitive:
            return validated_data.value
        return validated_data

    def cast(
        self, prompt: str, response_format: Type[Union[T, str, int, float, bool]]
    ) -> Any:
        """Simple chat completion for a single prompt with structured output"""
        return self._cast(messages=[user(prompt)], response_format=response_format)


class AgentOutput(BaseModel):
    internal_thoughts: str = Field(
        ...,
        description="Your first thoughts, this is a place to reflect and talk to yourself. This will not be shared with the group.",
    )
    response_to_last_communication: str = Field(
        ...,
        description="Direct response to the previous agent's message or instructions. Use this to acknowledge the prior message/instructions or to give feedback/corrections/comments to the prior agent.",
    )
    content: Optional[str] = Field(
        None,
        description="The actual work product or raw output (like written text, edits, etc.) - not instructions or coordination messages. Use this to pass on your work product to the next agent. Like you would do in real life sending a document to a colleague.",
    )
    stop_process: bool = Field(
        ...,
        description="IMPORTANT: Set to true when the process should end. Only authorized agents can set this to true. If this is set to true, the process will end and no more agents will be called.",
    )
    next_agent: str = Field(
        ..., description="The name of the next Agent to pass control to."
    )
    instructions_to_next_agent: str = Field(
        ..., description="Instructions passed on to the next agent."
    )


class AgentOutputWithMessage(AgentOutput):
    agent_message: str = Field(
        ..., description="A message to the user about the agent's output."
    )


class GroupMessage(BaseModel):
    agent_name: str = Field(
        ..., description="The name of the agent that sent the message."
    )
    content: str = Field(..., description="The content of the message.")


class Agent(LLM):
    """
    Represents a single agent with a specific role.
    """

    def __init__(
        self,
        name: str,
        role: str,
        *,
        model: str = "gpt-4o-mini",
    ):
        super().__init__(model)
        self.name = name
        self.role = role
        self.model = model
        self.messages: List[dict] = []
        self.system: str = None

    def chat(self, messages: List[dict] = []) -> AgentOutputWithMessage:
        """Chat completion with raw text response"""
        agent_output = self._cast(messages, AgentOutput)
        agent_message = f"=== {self.name} ===\n\n"
        agent_message += f"{self.name}'s response to last message:\n{agent_output.response_to_last_communication}\n\n"
        if agent_output.content:
            agent_message += f"{self.name}'s content:\n{agent_output.content}\n\n"
        agent_message += f"Next agent:\n{agent_output.next_agent}\n\n"
        agent_message += f"{self.name}'s message to {agent_output.next_agent}:\n{agent_output.instructions_to_next_agent}"

        return AgentOutputWithMessage(
            internal_thoughts=agent_output.internal_thoughts,
            response_to_last_communication=agent_output.response_to_last_communication,
            content=agent_output.content,
            stop_process=agent_output.stop_process,
            next_agent=agent_output.next_agent,
            instructions_to_next_agent=agent_output.instructions_to_next_agent,
            agent_message=agent_message,
        )


class Group:
    """
    Manages a list of Agents, orchestrates the conversation among them.
    """

    def __init__(
        self,
        agents: List[Agent],
        *,
        instructions: str = None,
        max_iterations: Optional[int] = None,
    ):
        self.agents = agents
        self.instructions = instructions
        self.messages: List[dict] = []
        self.max_iterations = max_iterations
        self.first_agent = self.agents[0]
        self.names_to_agents = {agent.name: agent for agent in self.agents}
        for agent in self.agents:
            agent.system = self.generate_agent_system(agent)

    def generate_group_information(self) -> str:
        """Generates a text including dynamic information about this specific group.
        Includes:
        - the names of the agents in the group
        - the roles of the agents in the group
        - the instructions for the group
        """
        name_to_role = {agent.name: agent.role for agent in self.agents}
        agent_names = ", ".join([agent.name for agent in self.agents])

        # Create dynamically numbered agent roles
        agent_roles = ""
        for i, agent in enumerate(self.agents, 1):
            agent_roles += f"{i}. {agent.name}\n"
            role_lines = name_to_role[agent.name].strip().split("\n")
            for line in role_lines:
                agent_roles += f"   {line.strip()}\n"
            agent_roles += "\n"

        return f"=== GROUP INFORMATION ===\n\nA. Group Members\n----------------\n{agent_names}\n\nB. Agent Roles \n--------------\n{agent_roles}\n\nC. Group Instructions\n---------------------\n{self.instructions}"

    def generate_agent_system(self, agent: Agent) -> str:
        """Generates the system prompt for a specific agent, it includes the group information and the agent's role."""
        instructions = f"""=== CORE INSTRUCTIONS ===\n\nYou are an agent, your name is '{agent.name}'.\nThis is who you are, this is your identity.\nYou are in a group with other agents. \nInformation about you, the group, objectives, and further instructions will be provided next."""
        agent_information = f"=== AGENT INFORMATION ===\n\nA. Agent Name\n---------\n{agent.name}\n\nB. Agent Role\n---------\n{agent.role}"
        group_information = self.generate_group_information()
        return f"{instructions}\n\n{agent_information}\n\n{group_information}"

    def run(self):
        n = 0
        group_messages: list[GroupMessage] = []
        stop_process = False
        working_agent = self.first_agent

        while not stop_process:
            n += 1
            print(f"=== {n} ===")
            print(f"=== {working_agent.name} ===")

            messages = [system(working_agent.system)]
            for group_message in group_messages:
                if group_message.agent_name == working_agent.name:
                    messages.append(assistant(group_message.content))
                else:
                    messages.append(user(group_message.content))

            print("--------------------------------")
            output: AgentOutputWithMessage = working_agent.chat(messages=messages)
            print(output.agent_message)
            group_messages.append(
                GroupMessage(
                    agent_name=working_agent.name, content=output.agent_message
                )
            )
            stop_process = output.stop_process
            if output.stop_process:
                print("=== STOPPING BY AGENT ===")

            if output.next_agent:
                working_agent = self.names_to_agents[output.next_agent]
            else:
                working_agent = self.first_agent

            if n > self.max_iterations:
                stop_process = True

        print("=== STOPPING ===")
        print(working_agent.name)
        print(messages)


if __name__ == "__main__":
    manager = Agent(
        name="Manager",
        role="""You are the manager, responsible ONLY for:
            1. Directing the workflow between writer and editor
            2. Delegating writing tasks to the writer
            3. Requesting reviews from the editor
            4. Making decisions based on editor's feedback
            You must NEVER write content yourself. Your job is purely coordination.""",
    )

    writer = Agent(
        name="Writer",
        role="""You are the writer. Your ONLY job is to write content when asked by the manager.
            You cannot make decisions about workflow or edit content.
            You must wait for specific writing instructions and then create that content.""",
    )

    editor = Agent(
        name="Editor",
        role="""You are the editor. Your ONLY job is to:
            1. Review content written by the writer
            2. Provide specific feedback and suggestions
            3. Give clear approval/rejection decisions
            You cannot write new content or direct the workflow.""",
    )

    # members will interact with each other, when a memmber is done, it will decide the next member to work on
    group = Group(
        [manager, writer, editor],
        max_iterations=20,
        instructions="""This group will create a book about fire following this strict workflow:
            1. Manager assigns specific writing tasks to writer
            2. Writer creates the content
            3. Manager sends content to editor for review
            4. Editor reviews and provides feedback
            5. If editor approves: they set stop_process=true
            6. If editor requests changes: process continues with manager
            Each member MUST stick to their role and follow this exact workflow.""",
    )

    group.run()
