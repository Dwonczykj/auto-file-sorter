# https://pub.towardsai.net/pydantic-ai-web-scraper-llama-3-3-python-powerful-ai-research-agent-6d634a9565fe
# Pydantic AI Web Scraper with Ollama and Streamlit AI Chatbot

from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import streamlit as st
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
import logging
import os
import asyncio
import datetime
from typing import Any
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

load_dotenv()

#  Ollama now has built-in compatibility with the OpenAI Chat Completions API, making it possible to use more tooling and applications with Ollama locally.
client = AsyncOpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required but not used,
    # api_key=os.getenv("OPENAI_API_KEY"),
)

model = OpenAIModel('llama2', openai_client=client)  # smaller model at 3.8GB
# model = OpenAIModel('llama3.3:latest', openai_client=client) # llama3.3:latest is 42GB too big

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("Please set TAVILY_API_KEY environment variable.")

tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str


@dataclass
class ResearchDependencies:
    todays_date: str


class ResearchResult(BaseModel):
    research_title: str = Field(
        description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(
        description='A main section that provides a detailed news article')
    research_bullets: str = Field(
        description='A set of bullet points summarizing key points')


# Create the agent
search_agent = Agent(
    model,
    deps_type=SearchDataclass,
    result_type=ResearchResult,
    system_prompt='You are a helpful research assistant, you are an expert in research. '
                  'When given a query, you will identify strong keywords to do 3-5 searches using the provided search tool. '
                  'Then combine results into a detailed response.'
)


@search_agent.system_prompt
async def add_current_date(ctx: RunContext[SearchDataclass]) -> str:
    todays_date = ctx.deps.todays_date
    system_prompt = (
        f"You're a helpful research assistant and an expert in research. "
        f"When given a question, write strong keywords to do 3-5 searches in total "
        f"(each with a query_number) and then combine the results. "
        f"If you need today's date it is {todays_date}. "
        f"Focus on providing accurate and current information."
    )
    return system_prompt


@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
    """Perform a search using the Tavily client."""
    max_results = search_data.deps.max_results
    results = await tavily_client.get_search_context(query=query, max_results=max_results)
    logging.info(f"Search results [{type(results)}]: {results}")
    # Convert results to dict if not already
    if not isinstance(results, dict):
        return {"results": str(results)}
    return results

# a main function


async def do_search(query: str, max_results: int):
    # Prepare dependencies
    current_date = datetime.date.today()
    date_string = current_date.strftime("%Y-%m-%d")
    deps = SearchDataclass(max_results=max_results, todays_date=date_string)
    result = await search_agent.run(query, deps=deps)
    return result


# streamlit app
def main3():
    st.title("Pydantic AI Web Scraper")
    query = st.text_input("Enter your query")
    if st.button("Search"):
        result = asyncio.run(do_search(query, max_results=5))
        st.write(result)


def main():
    st.set_page_config(page_title="AI News Researcher", layout="centered")

    st.title("Large Language Model News Researcher")
    st.write(
        "Stay updated on the latest trends and developments in Large Language Model.")

    # User input section
    st.sidebar.title("Search Parameters")
    query = st.sidebar.text_input(
        "Enter your query:", value="latest Large Language Model news")
    max_results = st.sidebar.slider(
        "Number of search results:", min_value=3, max_value=10, value=5)

    st.write("Use the sidebar to adjust search parameters.")

    if st.button("Get Latest Large Language Model News"):
        with st.spinner("Researching, please wait..."):
            result_data = asyncio.run(do_search(query, max_results))

        if not result_data.data:
            st.error("Failed to get research results")
            return

        st.markdown(result_data.data.research_title)
        # A bit of styling for the main article
        st.markdown(
            f"<div style='line-height:1.6;'>{result_data.data.research_main}</div>", unsafe_allow_html=True)

        st.markdown("### Key Takeaways")
        st.markdown(result_data.data.research_bullets)


if __name__ == "__main__":
    # streamlit run src/mains/pydantic_ai_streamlit_web_scraper.py
    main()
