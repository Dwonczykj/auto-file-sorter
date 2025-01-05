import os
from typing import TypeVar, Annotated
from pydantic import BaseModel
from pydantic_ai import Agent
from langgraph.graph import Graph, END
from typing import Dict, Any
from PIL import Image
from dotenv import load_dotenv


if not os.path.exists("images"):
    os.makedirs("images", exist_ok=True)
if not os.path.exists("auto_file_sorter/.env"):
    raise FileNotFoundError("auto_file_sorter/.env file not found")
load_dotenv("auto_file_sorter/.env")

assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"


class AssistantResponse(BaseModel):
    answer: str
    confidence: float

# More complex response type


class DetailedResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[str]
    metadata: dict[str, Any]

# Define state type for graph


class State(BaseModel):
    messages: list[str]
    current_response: AssistantResponse | None = None
    detailed_response: DetailedResponse | None = None
    summary: str | None = None


# Create agents with different types
assistant_agent = Agent(
    "openai:gpt-4",
    result_type=AssistantResponse,
    system_prompt="You are a helpful assistant. Always include a confidence score with your answers."
)
research_agent = Agent(
    "openai:gpt-4",
    result_type=DetailedResponse,
    system_prompt="You are a research assistant. Provide detailed answers with sources."
)
summary_agent = Agent(
    "openai:gpt-4",
    result_type=str,
    system_prompt="You are a summarization expert. Provide concise summaries."
)


async def assistant_node(state: State) -> Dict[str, Any]:
    """Node that uses our typed agent to process the input"""
    try:
        # Get the last message
        user_message = state.messages[-1]

        # Run the agent to get typed response
        result = await assistant_agent.run(user_message)

        # The result.data will be an AssistantResponse instance
        state.current_response = result.data

        # If confidence is too low, go to research
        if state.current_response.confidence < 0.5:
            return {"research": state}

        # Otherwise end
        return {"end": state}
    except Exception as e:
        print(f"Error in assistant node: {e}")
        return {"end": state}


async def research_node(state: State) -> Dict[str, Any]:
    """Node that performs detailed research"""
    try:
        result = await research_agent.run(state.messages[-1])
        state.detailed_response = result.data
        return {"summarize": state}
    except Exception as e:
        print(f"Error in research node: {e}")
        return {"end": state}


async def summary_node(state: State) -> Dict[str, Any]:
    """Node that summarizes research"""
    try:
        if state.detailed_response:
            result = await summary_agent.run(f"Summarize this: {state.detailed_response.answer}")
            state.summary = result.data
        return {"end": state}
    except Exception as e:
        print(f"Error in summary node: {e}")
        return {"end": state}

# Create the graph
workflow = Graph()

# Add nodes
workflow.add_node("assistant", assistant_node)
workflow.add_node("research", research_node)
workflow.add_node("summarize", summary_node)

# Add edges
workflow.add_edge("assistant", "research")
workflow.add_edge("assistant", END)
workflow.add_edge("research", "summarize")
workflow.add_edge("summarize", END)

# Compile the graph
app = workflow.compile()


with open("typed_agent_graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())
# display the image
Image.open("typed_agent_graph.png")


async def run_conversation():
    # Initialize state
    initial_state = State(
        messages=["What is the capital of France?"]
    )

    try:
        # Run the graph
        final_state = await app.ainvoke(initial_state)

        # Access the typed responses
        if final_state.current_response:
            print("\nDirect Answer:")
            print(f"Answer: {final_state.current_response.answer}")
            print(f"Confidence: {final_state.current_response.confidence}")

        if final_state.detailed_response:
            print("\nDetailed Research:")
            print(f"Answer: {final_state.detailed_response.answer}")
            print(f"Sources: {final_state.detailed_response.sources}")

        if final_state.summary:
            print("\nSummary:")
            print(final_state.summary)

    except Exception as e:
        print(f"Error running conversation: {e}")

# Run it
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_conversation())
