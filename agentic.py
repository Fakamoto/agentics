# Desired API of my Agentic LLM library
from agentic import Agent, Group
import json

manager = Agent(
    role="""You are the manager, responsible ONLY for:
        1. Directing the workflow between writer and editor
        2. Delegating writing tasks to the writer
        3. Requesting reviews from the editor
        4. Making decisions based on editor's feedback
        You must NEVER write content yourself. Your job is purely coordination."""
)

writer = Agent(
    role="""You are the writer. Your ONLY job is to write content when asked by the manager.
        You cannot make decisions about workflow or edit content.
        You must wait for specific writing instructions and then create that content."""
)

editor = Agent(
    role="""You are the editor. Your ONLY job is to:
        1. Review content written by the writer
        2. Provide specific feedback and suggestions
        3. Give clear approval/rejection decisions
        You cannot write new content or direct the workflow."""
)

# members will interact with each other, when a memmber is done, it will decide the next member to work on
group = Group(
    [manager, writer, editor],
    max_iterations=10,
    system="""This group will create a book about fire following this strict workflow:
         1. Manager assigns specific writing tasks to writer
         2. Writer creates the content
         3. Manager sends content to editor for review
         4. Editor reviews and provides feedback
         5. If editor approves: they set stop_process=true
         6. If editor requests changes: process continues with manager
         Each member MUST stick to their role and follow this exact workflow."""
)

group.run()

# After the group has finished running
final_text = group.query(
    agent=manager,
    message="Please give me the finished product"
)
print(final_text)