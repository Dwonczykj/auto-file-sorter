# # Agentic RAG: turbocharge your RAG with query reformulation and self-query! 🚀
# _Authored by: [Aymeric Roucher](https://huggingface.co/m-ric)_
#
# > This tutorial is advanced. You should have notions from [this other cookbook](advanced_rag) first!
#
# > Reminder: Retrieval-Augmented-Generation (RAG) is “using an LLM to answer a user query, but basing the answer on information retrieved from a knowledge base”. It has many advantages over using a vanilla or fine-tuned LLM: to name a few, it allows to ground the answer on true facts and reduce confabulations, it allows to provide the LLM with domain-specific knowledge, and it allows fine-grained control of access to information from the knowledge base.
#
# But vanilla RAG has limitations, most importantly these two:
# - It **performs only one retrieval step**: if the results are bad, the generation in turn will be bad.
# - __Semantic similarity is computed with the *user query* as a reference__, which might be suboptimal: for instance, the user query will often be a question and the document containing the true answer will be in affirmative voice, so its similarity score will be downgraded compared to other source documents in the interrogative form, leading to a risk of missing the relevant information.
#
# But we can alleviate these problems by making a **RAG agent: very simply, an agent armed with a retriever tool!**
#
# This agent will: ✅ Formulate the query itself and ✅ Critique to re-retrieve if needed.
#
# So it should naively recover some advanced RAG techniques!
# - Instead of directly using the user query as the reference in semantic search, the agent formulates itself a reference sentence that can be closer to the targeted documents, as in [HyDE](https://huggingface.co/papers/2212.10496)
# - The agent can the generated snippets and re-retrieve if needed, as in [Self-Query](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery/)
#
# Let's build this system. 🛠️
#
# Run the line below to install required dependencies:
import subprocess
subprocess.run(['pip', 'install', 'pandas', 'langchain', 'langchain-community', 'sentence-transformers', 'faiss-cpu', '"transformers[agents]"', '--upgrade', '-q'], check=True)

# Let's login in order to call the HF Inference API:
from huggingface_hub import notebook_login

notebook_login()
# We first load a knowledge base on which we want to perform RAG: this dataset is a compilation of the documentation pages for many `huggingface` packages, stored as markdown.
import datasets

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
# Now we prepare the knowledge base by processing the dataset and storing it into a vector database to be used by the retriever.
#
# We use [LangChain](https://python.langchain.com/) for its excellent vector database utilities.
# For the embedding model, we use [thenlper/gte-small](https://huggingface.co/thenlper/gte-small) since it performed well in our `RAG_evaluation` cookbook.
from tqdm import tqdm
from transformers import AutoTokenizer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

# Split docs and keep only unique ones
print("Splitting documents...")
docs_processed = []
unique_texts = {}
for doc in tqdm(source_docs):
    new_docs = text_splitter.split_documents([doc])
    for new_doc in new_docs:
        if new_doc.page_content not in unique_texts:
            unique_texts[new_doc.page_content] = True
            docs_processed.append(new_doc)

print(
    "Embedding documents... This should take a few minutes (5 minutes on MacBook with M1 Pro)"
)
embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
)
# Now the database is ready: let’s build our agentic RAG system!
#
# 👉 We only need a `RetrieverTool` that our agent can leverage to retrieve information from the knowledge base.
#
# Since we need to add a vectordb as an attribute of the tool, we cannot simply use the [simple tool constructor](https://huggingface.co/docs/transformers/main/en/agents#create-a-new-tool) with a `@tool` decorator: so we will follow the advanced setup highlighted in the [advanced agents documentation](https://huggingface.co/docs/transformers/main/en/agents_advanced#directly-define-a-tool-by-subclassing-tool-and-share-it-to-the-hub).
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        return "\nRetrieved documents:\n" + "".join(
            [
                f"===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
# Now it’s straightforward to create an agent that leverages this tool!
#
# The agent will need these arguments upon initialization:
# - *`tools`*: a list of tools that the agent will be able to call.
# - *`llm_engine`*: the LLM that powers the agent.
#
# Our `llm_engine` must be a callable that takes as input a list of [messages](https://huggingface.co/docs/transformers/main/chat_templating) and returns text. It also needs to accept a `stop_sequences` argument that indicates when to stop its generation. For convenience, we directly use the `HfEngine` class provided in the package to get a LLM engine that calls our [Inference API](https://huggingface.co/docs/api-inference/en/index).
#
# And we use [meta-llama/Llama-3.1-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) as the llm engine because:
# - It has a long 128k context, which is helpful for processing long source documents
# - It is served for free at all times on HF's Inference API!
#
# _Note:_ The Inference API hosts models based on various criteria, and deployed models may be updated or replaced without prior notice. Learn more about it [here](https://huggingface.co/docs/api-inference/supported-models).
from transformers.agents import HfApiEngine, ReactJsonAgent

llm_engine = HfApiEngine("Qwen/Qwen2.5-72B-Instruct")

retriever_tool = RetrieverTool(vectordb)
agent = ReactJsonAgent(
    tools=[retriever_tool], llm_engine=llm_engine, max_iterations=4, verbose=2
)
# Since we initialized the agent as a `ReactJsonAgent`, it has been automatically given a default system prompt that tells the LLM engine to process step-by-step and generate tool calls as JSON blobs (you could replace this prompt template with your own as needed).
#
# Then when its `.run()` method is launched, the agent takes care of calling the LLM engine, parsing the tool call JSON blobs and executing these tool calls, all in a loop that ends only when the final answer is provided.
agent_output = agent.run("How can I push a model to the Hub?")

print("Final output:")
print(agent_output)
# ## Agentic RAG vs. standard RAG
#
# Does the agent setup make a better RAG system? Well, let's compare it to a standard RAG system using LLM Judge!
#
# We will use [meta-llama/Meta-Llama-3-70B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) for evaluation since it's one of the strongest OS models we tested for LLM judge use cases.
eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
# Before running the test let's make the agent less verbose.
import logging

agent.logger.setLevel(logging.WARNING) # Let's reduce the agent's verbosity level

eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train")
outputs_agentic_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]

    enhanced_question = f"""Using the information contained in your knowledge base, which you can access with the 'retriever' tool,
give a comprehensive answer to the question below.
Respond only to the question asked, response should be concise and relevant to the question.
If you cannot find information, do not give up and try calling your retriever again with different arguments!
Make sure to have covered the question completely by calling the retriever tool several times with semantically different queries.
Your queries should not be questions but affirmative form sentences: e.g. rather than "How do I load a model from the Hub in bf16?", query should be "load a model from the Hub bf16 weights".

Question:
{question}"""
    answer = agent.run(enhanced_question)
    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_agentic_rag.append(results_agentic)
from huggingface_hub import InferenceClient

reader_llm = InferenceClient("Qwen/Qwen2.5-72B-Instruct")

outputs_standard_rag = []

for example in tqdm(eval_dataset):
    question = example["question"]
    context = retriever_tool(question)

    prompt = f"""Given the question and supporting documents below, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If you cannot find information, do not give up and try calling your retriever again with different arguments!

Question:
{question}

{context}
"""
    messages = [{"role": "user", "content": prompt}]
    answer = reader_llm.chat_completion(messages).choices[0].message.content

    print("=======================================================")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print(f'True answer: {example["answer"]}')

    results_agentic = {
        "question": question,
        "true_answer": example["answer"],
        "source_doc": example["source_doc"],
        "generated_answer": answer,
    }
    outputs_standard_rag.append(results_agentic)
# The evaluation prompt follows some of the best principles shown in [our llm_judge cookbook](llm_judge): it follows a small integer Likert scale, has clear criteria, and a description for each score.
EVALUATION_PROMPT = """You are a fair evaluator language model.

You will be given an instruction, a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 3. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 3}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.
5. Do not score conciseness: a correct answer that covers the question should receive max score, even if it contains additional useless information.

The instruction to evaluate:
{instruction}

Response to evaluate:
{response}

Reference Answer (Score 3):
{reference_answer}

Score Rubrics:
[Is the response complete, accurate, and factual based on the reference answer?]
Score 1: The response is completely incomplete, inaccurate, and/or not factual.
Score 2: The response is somewhat complete, accurate, and/or factual.
Score 3: The response is completely complete, accurate, and/or factual.

Feedback:"""
from huggingface_hub import InferenceClient

evaluation_client = InferenceClient("meta-llama/Llama-3.1-70B-Instruct")
import pandas as pd

results = {}
for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:
    for experiment in tqdm(outputs):
        eval_prompt = EVALUATION_PROMPT.format(
            instruction=experiment["question"],
            response=experiment["generated_answer"],
            reference_answer=experiment["true_answer"],
        )
        messages = [
            {"role": "system", "content": "You are a fair evaluator language model."},
            {"role": "user", "content": eval_prompt},
        ]

        eval_result = evaluation_client.text_generation(
            eval_prompt, max_new_tokens=1000
        )
        try:
            feedback, score = [item.strip() for item in eval_result.split("[RESULT]")]
            experiment["eval_score_LLM_judge"] = score
            experiment["eval_feedback_LLM_judge"] = feedback
        except:
            print(f"Parsing failed - output was: {eval_result}")

    results[system_type] = pd.DataFrame.from_dict(outputs)
    results[system_type] = results[system_type].loc[~results[system_type]["generated_answer"].str.contains("Error")]
DEFAULT_SCORE = 2 # Give average score whenever scoring fails
def fill_score(x):
    try:
        return int(x)
    except:
        return DEFAULT_SCORE

for system_type, outputs in [
    ("agentic", outputs_agentic_rag),
    ("standard", outputs_standard_rag),
]:

    results[system_type]["eval_score_LLM_judge_int"] = (
        results[system_type]["eval_score_LLM_judge"].fillna(DEFAULT_SCORE).apply(fill_score)
    )
    results[system_type]["eval_score_LLM_judge_int"] = (results[system_type]["eval_score_LLM_judge_int"] - 1) / 2

    print(
        f"Average score for {system_type} RAG: {results[system_type]['eval_score_LLM_judge_int'].mean()*100:.1f}%"
    )
# **Let us recap: the Agent setup improves scores by 14% compared to a standard RAG!** (from 73.1% to 86.9%)
#
# This is a great improvement, with a very simple setup 🚀
#
# (For a baseline, using Llama-3-70B without the knowledge base got 36%)
