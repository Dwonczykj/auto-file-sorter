from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

# Define Pydantic models for input/output


class UserQuestion(BaseModel):
    question: str = Field(description="The user's question")
    context: Optional[str] = Field(
        default=None, description="Optional context for the question")


class AnalysisResult(BaseModel):
    main_points: List[str] = Field(
        description="Key points extracted from the analysis")
    conclusion: str = Field(description="Final conclusion")
    confidence_score: float = Field(
        description="Confidence score between 0 and 1")

# Create a simple agent that uses Pydantic models


class PydanticAgent:
    """
    This is a simple example of how to use Pydantic models with Langchain and Langgraph.
    It is a simple agent that uses a LLM via langchain API to analyze a user's question and provide a structured response using the Pydantic Output Parser.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.output_parser = PydanticOutputParser(
            pydantic_object=AnalysisResult)

        # Create a prompt template that includes message history
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an analytical assistant. Analyze the user's question and provide a structured response. "
                "Your response should be formatted as a JSON object with the following structure:\n"
                "{format_instructions}"
            )),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="Question: {question}\nContext: {context}")
        ])

    def analyze(self,
                question: str,
                context: Optional[str] = None,
                history: Optional[List] = None) -> AnalysisResult:
        # Format the prompt with the parser instructions
        formatted_prompt = self.prompt.partial(
            format_instructions=self.output_parser.get_format_instructions()
        )

        # Prepare the messages history
        messages_history = history if history else []

        # Create the chain
        chain = formatted_prompt | self.llm | self.output_parser

        # Execute the chain
        result = chain.invoke({
            "question": question,
            "context": context or "",
            "history": messages_history
        })

        return result

# Function to create a simple graph


def create_analysis_graph():
    # Define the initial state
    def initial_state():
        return {
            "messages": [],
            "current_analysis": None
        }

    # Create the graph
    graph = StateGraph(initial_state=initial_state)

    # Create agent instance
    agent = PydanticAgent()

    # Define the analysis node
    def analyze(state):
        # Get the latest message
        current_message = state["messages"][-1]

        # Perform analysis
        result = agent.analyze(
            question=current_message.content,
            history=state["messages"][:-1]
        )

        # Update state
        return {
            "messages": state["messages"],
            "current_analysis": result
        }

    # Add node to graph
    graph.add_node("analyze", analyze)

    # Add edges
    graph.add_edge("analyze", END)

    # Set entry point
    graph.set_entry_point("analyze")

    return graph.compile()

# Example usage


def main():
    # Create input
    user_input = UserQuestion(
        question="What are the main challenges in implementing AI systems?",
        context="Focus on ethical considerations and technical limitations."
    )

    # Create and run the graph
    graph = create_analysis_graph()

    # Run the graph
    result = graph.invoke({
        "messages": [
            HumanMessage(content=user_input.question)
        ]
    })

    # Access the analysis result
    analysis = result["current_analysis"]
    print(f"Main Points: {analysis.main_points}")
    print(f"Conclusion: {analysis.conclusion}")
    print(f"Confidence Score: {analysis.confidence_score}")


if __name__ == "__main__":
    main()
