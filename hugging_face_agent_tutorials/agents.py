# # Build an agent with tool-calling superpowers 🦸 using Transformers Agents
# _Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_
#
# This notebook demonstrates how you can use [**Transformers Agents**](https://huggingface.co/docs/transformers/en/agents) to build awesome **agents**!
#
# What are **agents**? Agents are systems that are powered by an LLM and enable the LLM (with careful prompting and output parsing) to use specific *tools* to solve problems.
#
# These *tools* are basically functions that the LLM couldn't perform well by itself: for instance for a text-generation LLM like [Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct), this could be an image generation tool, a web search tool, a calculator...
#
# What is **Transformers Agents**? it's an extension of our `transformers` library that provides building blocks to build your own agents! Learn more about it in the [documentation](https://huggingface.co/docs/transformers/en/agents).
#
# Let's see how to use it, and which use cases it can solve.
#
# Run the line below to install required dependencies:
import subprocess
subprocess.run(['pip', 'install', '"transformers[agents]"', 'datasets', 'langchain', 'sentence-transformers', 'faiss-cpu', 'duckduckgo-search', 'openai', 'langchain-community', '--upgrade', '-q'], check=True)

# Let's login in order to call the HF Inference API:
from huggingface_hub import notebook_login

notebook_login()
# ## 1. 🏞️ Multimodal + 🌐 Web-browsing assistant
#
# For this use case, we want to show an agent that browses the web and is able to generate images.
#
# To build it, we simply need to have two tools ready: image generation and web search.
# - For image generation, we load a tool from the Hub that uses the HF Inference API (Serverless) to generate images using Stable Diffusion.
# - For the web search, we use a built-in tool.
from transformers import load_tool, ReactCodeAgent, HfApiEngine

# Import tool from Hub
image_generation_tool = load_tool("m-ric/text-to-image", cache=False)

# Import tool from LangChain
from transformers.agents.search import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")
# Initialize the agent with both tools
agent = ReactCodeAgent(
    tools=[image_generation_tool, search_tool], llm_engine=llm_engine
)

# Run it!
result = agent.run(
    "Generate me a photo of the car that James bond drove in the latest movie.",
)
result
# Image: Image of an Aston Martin DB5

# ## 2. 📚💬 RAG with Iterative query refinement & Source selection
#
# Quick definition: Retrieval-Augmented-Generation (RAG) is ___“using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base”.___
#
# This method has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.
#
# - Now let’s say we want to perform RAG, but with the additional constraint that some parameters must be dynamically generated. For example, depending on the user query we could want to restrict the search to specific subsets of the knowledge base, or we could want to adjust the number of documents retrieved. The difficulty is: **how to dynamically adjust these parameters based on the user query?**
#
# - A frequent failure case of RAG is when the retrieval based on the user query does not return any relevant supporting documents. **Is there a way to iterate by re-calling the retriever with a modified query in case the previous results were not relevant?**
#
#
# 🔧 Well, we can solve the points above in a simple way: we will **give our agent control over the retriever's parameters!**
#
# ➡️ Let's show how to do this. We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many `huggingface` packages, stored as markdown.

import datasets

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
# Now we prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever. We are going to use LangChain, since it features excellent utilities for vector databases:

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(
    source_docs
)[:1000]

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(documents=docs_processed, embedding=embedding_model)
# Now that we have the database ready, let’s build a RAG system that answers user queries based on it!
#
# We want our system to select only from the most relevant sources of information, depending on the query.
#
# Our documentation pages come from the following sources:
all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))
print(all_sources)
# 👉 Now let's build a `RetrieverTool` that our agent can leverage to retrieve information from the knowledge base.
#
# Since we need to add a vectordb as an attribute of the tool, we cannot simply use the [simple tool constructor](https://huggingface.co/docs/transformers/main/en/agents#create-a-new-tool) with a `@tool` decorator: so we will follow the advanced setup highlighted in the [advanced agents documentation](https://huggingface.co/docs/transformers/main/en/agents_advanced#directly-define-a-tool-by-subclassing-tool-and-share-it-to-the-hub).
import json
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "source": {"type": "string", "description": ""},
        "number_of_documents": {
            "type": "string",
            "description": "the number of documents to retrieve. Stay under 10 to avoid drowning in docs",
        },
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, all_sources: str, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.inputs["source"]["description"] = (
            f"The source of the documents to search, as a str representation of a list. Possible values in the list are: {all_sources}. If this argument is not provided, all sources will be searched.".replace(
                "'", "`"
            )
        )

    def forward(self, query: str, source: str = None, number_of_documents=7) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        number_of_documents = int(number_of_documents)

        if source:
            if isinstance(source, str) and "[" not in str(
                source
            ):  # if the source is not representing a list
                source = [source]
            source = json.loads(str(source).replace("'", '"'))

        docs = self.vectordb.similarity_search(
            query,
            filter=({"source": source} if source else None),
            k=number_of_documents,
        )

        if len(docs) == 0:
            return "No documents found with this filtering. Try removing the source filter."
        return "Retrieved documents:\n\n" + "\n===Document===\n".join(
            [doc.page_content for doc in docs]
        )
# ### Optional: Share your Retriever tool to Hub
#
# To share your tool to the Hub, first copy-paste the code in the RetrieverTool definition cell to a new file named for instance `retriever.py`.
#
# When the tool is loaded from a separate file, you can then push it to the Hub using the code below (make sure to login with a `write` access token)
share_to_hub = True

if share_to_hub:
    from huggingface_hub import login
    from retriever import RetrieverTool

    login("your_token")

    tool = RetrieverTool(vectordb, all_sources)

    tool.push_to_hub(repo_id="m-ric/retriever-tool")

    # Loading the tool
    from transformers.agents import load_tool

    retriever_tool = load_tool(
        "m-ric/retriever-tool", vectordb=vectordb, all_sources=all_sources
    )
# ### Run the agent!
from transformers.agents import HfApiEngine, ReactJsonAgent

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

retriever_tool = RetrieverTool(vectordb=vectordb, all_sources=all_sources)
agent = ReactJsonAgent(tools=[retriever_tool], llm_engine=llm_engine, verbose=0)

agent_output = agent.run("Please show me a LORA finetuning script")

print("Final output:")
print(agent_output)
# What happened here? First, the agent launched the retriever with specific sources in mind (`['transformers', 'blog']`).
#
# But this retrieval did not yield enough results ⇒ no problem! The agent could iterate on previous results, so it just re-ran its retrieval with less restrictive search parameters.
# Thus the research was successful!
#
# Note that **using an LLM agent** that calls a retriever as a tool and can dynamically modify the query and other retrieval parameters **is a more general formulation of RAG**, which also covers many RAG improvement techniques like iterative query refinement.
#
# ## 3. 💻 Debug Python code
# Since the ReactCodeAgent has a built-in Python code interpreter, we can use it to debug our faulty Python script!
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], llm_engine=HfApiEngine("Qwen/Qwen2.5-72B-Instruct"))

code = """
list=[0, 1, 2]

for i in range(4):
    print(list(i))
"""

final_answer = agent.run(
    "I have some code that creates a bug: please debug it, then run it to make sure it works and return the final code",
    code=code,
)
# As you can see, the agent tried the given code, gets an error, analyses the error, corrects the code and returns it after veryfing that it works!
#
# And the final code is the corrected code:
print(final_answer)
# ## 4. Create your own LLM engine (OpenAI)
#
# It's really easy to set up your own LLM engine:
# it only needs a `__call__` method with these criteria:
# 1. Takes as input a list of messages in [ChatML format](https://huggingface.co/docs/transformers/main/en/chat_templating#introduction) and outputs the answer.
# 2. Accepts a `stop_sequences` arguments to pass sequences on which generation stops.
# 3. Depending on which kind of message roles your LLM accepts, you may also need to convert some message roles.
import os
from openai import OpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: "user",
}


class OpenAIEngine:
    def __init__(self, model_name="gpt-4o-2024-05-13"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        # Get clean message list
        messages = get_clean_message_list(
            messages, role_conversions=openai_role_conversions
        )

        # Get LLM output
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
        )
        return response.choices[0].message.content


openai_engine = OpenAIEngine()
agent = ReactCodeAgent(llm_engine=openai_engine, tools=[])

code = """
list=[0, 1, 2]

for i in range(4):
    print(list(i))
"""

final_answer = agent.run(
    "I have some code that creates a bug: please debug it and return the final code",
    code=code,
)
print(final_answer)
# ## ➡️ Conclusion
#
# The use cases above should give you a glimpse into the possibilities of our Agents framework!
#
# For more advanced usage, read the [documentation](https://huggingface.co/docs/transformers/en/transformers_agents), and [this experiment](https://github.com/aymeric-roucher/agent_reasoning_benchmark/blob/main/benchmark_gaia.ipynb) that allowed us to build our own agent based on Llama-3-70B that beats many GPT-4 agents on the very difficult [GAIA Leaderboard](https://huggingface.co/spaces/gaia-benchmark/leaderboard)!
#
# All feedback is welcome, it will help us improve the framework! 🚀
