# # Have several agents collaborate in a multi-agent hierarchy 🤖🤝🤖
# _Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_
#
# > This tutorial is advanced. You should have notions from [this other cookbook](agents) first!
#
# In this notebook we will make a **multi-agent web browser: an agentic system with several agents collaborating to solve problems using the web!**
#
# It will be a simple hierarchy, using a `ManagedAgent` object to wrap the managed web search agent:
#
# ```
#               +----------------+
#               | Manager agent  |
#               +----------------+
#                        |
#         _______________|______________
#        |                              |
#   Code interpreter   +--------------------------------+
#        tool          |         Managed agent          |
#                     |      +------------------+      |
#                     |      | Web Search agent |      |
#                     |      +------------------+      |
#                     |         |            |         |
#                     |  Web Search tool     |         |
#                     |             Visit webpage tool |
#                     +--------------------------------+
# ```
# Let's set up this system.
#
# Run the line below to install the required dependencies:

from transformers.agents.search import DuckDuckGoSearchTool
from transformers.agents import (
    ReactCodeAgent,
    ReactJsonAgent,
    HfApiEngine,
    ManagedAgent,
)
from transformers.agents import tool
from requests.exceptions import RequestException
from markdownify import markdownify as md
import requests
import re
from huggingface_hub import notebook_login
import subprocess
import sys


def install_dependencies():
    """Install required packages if not already installed"""
    packages = ["markdownify", "duckduckgo-search", "transformers[agents]"]
    for package in packages:
        subprocess.run([sys.executable, "-m", "pip",
                       "install", package, "--upgrade", "-q"])


# Only run installation if this is the main script
if __name__ == "__main__":
    install_dependencies()

# Let's login in order to call the HF Inference API:


notebook_login()

# ⚡️ Our agent will be powered by [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) using `HfApiEngine` class that uses HF's Inference API: the Inference API allows to quickly and easily run any OS model.
#
# _Note:_ The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it [here](https://huggingface.co/docs/api-inference/supported-models).

model = "Qwen/Qwen2.5-72B-Instruct"

# ### 🔍 Create a web search tool
#
# For web browsing, we can already use our pre-existing [`DuckDuckGoSearchTool`](https://github.com/huggingface/transformers/blob/main/src/transformers/agents/search.py) tool to provide a Google search equivalent.
#
# But then we will also need to be able to peak into the page found by the `DuckDuckGoSearchTool`.
# To do so, we could import the library's built-in `VisitWebpageTool`, but we will build it again to see how it's done.
#
# So let's create our `VisitWebpageTool` tool from scratch using `markdownify`.


@tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given URL and returns its content as a markdown string.

    Args:
        url: The URL of the webpage to visit.

    Returns:
        The content of the webpage converted to Markdown, or an error message if the request fails.
    """
    try:
        # Send a GET request to the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Convert the HTML content to Markdown
        markdown_content = md(response.text).strip()

        # Remove multiple line breaks
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        return markdown_content

    except RequestException as e:
        return f"Error fetching the webpage: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

# Ok, now let's initialize and test our tool!


print(visit_webpage("https://en.wikipedia.org/wiki/Hugging_Face")[:500])

# ## Build our multi-agent system 🤖🤝🤖
#
# Now that we have all the tools `search` and `visit_webpage`, we can use them to create the web agent.
#
# Which configuration to choose for this agent?
# - Web browsing is a single-timeline task that does not require parallel tool calls, so JSON tool calling works well for that. We thus choose a `ReactJsonAgent`.
# - Also, since sometimes web search requires exploring many pages before finding the correct answer, we prefer to increase the number of `max_iterations` to 10.


llm_engine = HfApiEngine(model)

web_agent = ReactJsonAgent(
    tools=[DuckDuckGoSearchTool(), visit_webpage],
    llm_engine=llm_engine,
    max_iterations=10,
)

# We then wrap this agent into a `ManagedAgent` that will make it callable by its manager agent.

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

# Finally we create a manager agent, and upon initialization we pass our managed agent to it in its `managed_agents` argument.
#
# Since this agent is the one tasked with the planning and thinking, advanced reasoning will be beneficial, so a `ReactCodeAgent` will be the best choice.
#
# Also, we want to ask a question that involves the current year: so let us add `additional_authorized_imports=["time", "datetime"]`

manager_agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "datetime"],
)

# That's all! Now let's run our system! We select a question that requires some calculation and

manager_agent.run("How many years ago was Stripe founded?")

# Our agents managed to efficiently collaborate towards solving the task! ✅
#
# 💡 You can easily extend this to more agents: one does the code execution, one the web search, one handles file loadings...
#
# 🤔💭 One could even think of doing more complex, tree-like hierarchies, with one CEO agent handling multiple middle managers, each with several reports.
#
# We could even add more intermediate layers of management, each with multiple daily meetings, lots of agile stuff with scrum masters, and each new component adds enough friction to ensure the tasks never get done... Ehm wait, no, let's stick with our simple structure.
