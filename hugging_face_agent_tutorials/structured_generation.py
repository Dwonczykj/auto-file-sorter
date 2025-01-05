# # RAG with source highlighting using Structured generation
# _Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_
#
# **Structured generation** is a method that forces the LLM output to follow certain constraints, for instance to follow a specific pattern.
#
# This has numerous use cases:
# - ✅ Output a dictionary with specific keys
# - 📏 Make sure the output will be longer than N characters
# - ⚙️ More generally, force the output to follow a certain regex pattern for downtream processing.
# - 💡 Highlight sources supporting the answer in Retrieval-Augmented-Generation (RAG)
#
#
# In this notebook, we demonstrate specifically the last use case:
#
# **➡️ We build a RAG system that not only provides an answer, but also highlights the supporting snippets that this answer is based on.**
#
# _If you need an introduction to RAG, you can check out [this other cookbook](advanced_rag)._
#
# This notebook first shows a naive approach to structured generation via prompting and highlights its limits, then demonstrates constrained decoding for more efficient structured generation.
#
# It leverages HuggingFace Inference Endpoints (the example shows a [serverless](https://huggingface.co/docs/api-inference/quicktour) endpoint, but you can directly change the endpoint to a [dedicated](https://huggingface.co/docs/inference-endpoints/en/guides/access) one), then also shows a local pipeline using [outlines](https://github.com/outlines-dev/outlines), a structured text generation library.
import subprocess
subprocess.run(['pip', 'install', 'pandas', 'json', 'huggingface_hub', 'pydantic', 'outlines', 'accelerate', '-q'], check=True)

import pandas as pd
import json
from huggingface_hub import InferenceClient

pd.set_option("display.max_colwidth", None)
repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

llm_client = InferenceClient(model=repo_id, timeout=120)

# Test your LLM client
llm_client.text_generation(prompt="How are you today?", max_new_tokens=20)
# ## Prompting the model
#
# To get structured outputs from your model, you can simply prompt a powerful enough models with appropriate guidelines, and it should work directly... most of the time.
#
# In this case, we want the RAG model to generate not only an answer, but also a confidence score and some source snippets.
# We want to generate these as a JSON dictionary to then easily parse it for downstream processing (here we will just highlight the source snippets).
RELEVANT_CONTEXT = """
Document:

The weather is really nice in Paris today.
To define a stop sequence in Transformers, you should pass the stop_sequence argument in your pipeline or model.

"""
RAG_PROMPT_TEMPLATE_JSON = """
Answer the user query based on the source documents.

Here are the source documents: {context}


You should provide your answer as a JSON blob, and also provide all relevant short source snippets from the documents on which you directly based your answer, and a confidence score as a float between 0 and 1.
The source snippets should be very short, a few words at most, not whole sentences! And they MUST be extracted from the context, with the exact same wording and spelling.

Your answer should be built as follows, it must contain the "Answer:" and "End of answer." sequences.

Answer:
{{
  "answer": your_answer,
  "confidence_score": your_confidence_score,
  "source_snippets": ["snippet_1", "snippet_2", ...]
}}
End of answer.

Now begin!
Here is the user question: {user_query}.
Answer:
"""
USER_QUERY = "How can I define a stop sequence in Transformers?"
prompt = RAG_PROMPT_TEMPLATE_JSON.format(
    context=RELEVANT_CONTEXT, user_query=USER_QUERY
)
print(prompt)
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=1000,
)

answer = answer.split("End of answer.")[0]
print(answer)
# The output of the LLM is a string representation of a dictionary: so let's just load it as a dictionary using `literal_eval`.
from ast import literal_eval

parsed_answer = literal_eval(answer)
def highlight(s):
    return "\x1b[1;32m" + s + "\x1b[0m"


def print_results(answer, source_text, highlight_snippets):
    print("Answer:", highlight(answer))
    print("\n\n", "=" * 10 + " Source documents " + "=" * 10)
    for snippet in highlight_snippets:
        source_text = source_text.replace(snippet.strip(), highlight(snippet.strip()))
    print(source_text)


print_results(
    parsed_answer["answer"], RELEVANT_CONTEXT, parsed_answer["source_snippets"]
)
# This works! 🥳
#
# But what about using a less powerful model?
#
# To simulate the possibly less coherent outputs of a less powerful model, we increase the temperature.
answer = llm_client.text_generation(
    prompt,
    max_new_tokens=250,
    temperature=1.6,
    return_full_text=False,
)
print(answer)
# Now, the output is not even in correct JSON.
# ## 👉 Constrained decoding
#
# To force a JSON output, we'll have to use **constrained decoding** where we force the LLM to only output tokens that conform to a set of rules called a **grammar**.
#
# This grammar can be defined using Pydantic models, JSON schema, or regular expressions. The AI will then generate a response that conforms to the specified grammar.
#
# Here for instance we follow [Pydantic types](https://docs.pydantic.dev/latest/api/types/).
from pydantic import BaseModel, confloat, StringConstraints
from typing import List, Annotated


class AnswerWithSnippets(BaseModel):
    answer: Annotated[str, StringConstraints(min_length=10, max_length=100)]
    confidence: Annotated[float, confloat(ge=0.0, le=1.0)]
    source_snippets: List[Annotated[str, StringConstraints(max_length=30)]]
# I advise inspecting the generated schema to check that it correctly represents your requirements:
AnswerWithSnippets.schema()
# You can use either the client's `text_generation` method or use its `post` method.
# Using text_generation
answer = llm_client.text_generation(
    prompt,
    grammar={"type": "json", "value": AnswerWithSnippets.schema()},
    max_new_tokens=250,
    temperature=1.6,
    return_full_text=False,
)
print(answer)

# Using post
data = {
    "inputs": prompt,
    "parameters": {
        "temperature": 1.6,
        "return_full_text": False,
        "grammar": {"type": "json", "value": AnswerWithSnippets.schema()},
        "max_new_tokens": 250,
    },
}
answer = json.loads(llm_client.post(json=data))[0]["generated_text"]
print(answer)
# ✅ Although the answer is still nonsensical due to the high temperature, the generated output is now correct JSON format, with the exact keys and types we defined in our grammar!
#
# It can then be parsed for further processing.
# ### Grammar on a local pipeline with Outlines
#
# [Outlines](https://github.com/outlines-dev/outlines/) is the library that runs under the hood on our Inference API to constrain output generation. You can also use it locally.
#
# It works by [applying a bias on the logits](https://github.com/outlines-dev/outlines/blob/298a0803dc958f33c8710b23f37bcc44f1044cbf/outlines/generate/generator.py#L143) to force selection of only the ones that conform to your constraint.
import outlines

repo_id = "mustafaaljadery/gemma-2B-10M"
# Load model locally
model = outlines.models.transformers(repo_id)

schema_as_str = json.dumps(AnswerWithSnippets.schema())

generator = outlines.generate.json(model, schema_as_str)

# Use the `generator` to sample an output from the model
result = generator(prompt)
print(result)
# You can also use [Text-Generation-Inference](https://huggingface.co/docs/text-generation-inference/en/index) with constrained generation (see the [documentation](https://huggingface.co/docs/text-generation-inference/en/conceptual/guidance) for more details and examples).
#
# Now we've demonstrated a specific RAG use-case, but constrained generation is helpful for much more than that.
#
# For instance in your [LLM judge](llm_judge) workflows, you can also use constrained generation to output a JSON, as follows:
# ```
# {
#     "score": 1,
#     "rationale": "The answer does not match the true answer at all."
#     "confidence_level": 0.85
# }
# ```
# That's all for today, congrats for following along! 👏
