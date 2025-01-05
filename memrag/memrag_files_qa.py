from memorag import MemoRAG
import requests
import tiktoken
import tarfile
import os
import time
from typing import TypedDict

install_cmd = """
pip install memorag==0.1.3
pip install faiss-gpu # please install faiss using conda to obtain the latest version. Here using pip as example
pip install flash_attn
pip install -U bitsandbytes
"""


class DebugHPTokenLen(TypedDict):
    small_part: str
    content: str


def debug_hp_token_len() -> DebugHPTokenLen:
    encoding = tiktoken.get_encoding("cl100k_base")

    url = 'https://raw.githubusercontent.com/qhjqhj00/MemoRAG/main/examples/harry_potter.txt'
    response = requests.get(url)
    content = response.text

    print(f"The raw database has {len(encoding.encode(content))} tokens...")

    small_part = " ".join(content.split()[:50000])
    print(f"Using part of the database: with {
          len(encoding.encode(small_part))} tokens...")
    return DebugHPTokenLen(small_part=small_part, content=content)


def download_chunks_url(
    url: str,
    download_path: str,
    extract_path: str
):

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"File downloaded successfully: {download_path}")
    else:
        print(f"Failed to download file: {response.status_code}")

    if os.path.exists(download_path):
        with tarfile.open(download_path, 'r:bz2') as tar:
            tar.extractall(path=extract_path)
        print(f"File extracted successfully to: {extract_path}")
    else:
        print("Downloaded file not found.")


def load_weights_from_disk(pipe: MemoRAG):
    start = time.time()
    pipe.load("/content/harry_potter_qwen2_ratio16", print_stats=True)
    print(f"Loading from cache takes {
          round(time.time()-start, 2)} for the full book.")


def get_pipe():

    # Initialize MemoRAG pipeline
    pipe = MemoRAG(
        mem_model_name_or_path="TommyChien/memorag-mistral-7b-inst",
        ret_model_name_or_path="BAAI/bge-m3",
        # Optional: if not specify, use memery model as the generator
        gen_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
        cache_dir="path_to_model_cache",  # Optional: specify local model cache directory
        access_token="hugging_face_access_token",  # Optional: Hugging Face access token
        beacon_ratio=4
    )
    # pipe = MemoRAG(
    #     mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
    #     ret_model_name_or_path="BAAI/bge-m3",
    #     beacon_ratio=16,
    #     load_in_4bit=True,
    #     enable_flash_attn=False # T4 GPU does not support flash attention when running on google cloud
    # )
    return pipe


def run_pipe(pipe: MemoRAG):
    context = open("examples/harry_potter.txt").read()
    query = "How many times is the Chamber of Secrets opened in the book?"

    # Memorize the context and save to cache
    pipe.memorize(context, save_dir="cache/harry_potter/", print_stats=True)
    # Through the MemoRAG.memorize() method, the memory model can build global memory in a longer input context.
    # By adjusting the parameter beacon_ratio, the model’s ability to handle longer contexts can be extended.

    # Generate response using the memorized context
    res = pipe(context=context, query=query,
               task_type="memorag", max_new_tokens=256)
    print(f"MemoRAG generated answer: \n{res}")


def qa_hp_files(
    pipe: MemoRAG,
    res: DebugHPTokenLen
):
    # perform QA task
    # Currently, MemoRAG primarily focuses on two key tasks: question-answering (QA) and summarization.

    query = "What's the theme of the book?"

    result = pipe(context=res["small_part"], query=query,
                  task_type="qa", max_new_tokens=256)
    print(f"Using memory to produce the answer: \n{result} \n\n")
    result = pipe(context=res["small_part"], query=query,
                  task_type="memorag", max_new_tokens=256)
    print(f"Using MemoRAG to produce the answer: \n{result[0]}")

    # perform retrieval task

    clues = pipe.mem_model.rewrite(query).split("\n")
    # Filter out short or irrelevant clues
    clues = [q for q in clues if len(q.split()) > 3]
    print("Clues generated from memory:\n", clues)

    # Retrieve relevant passages based on the recalled clues
    retrieved_passages = pipe._retrieve(clues)
    print("Retrieved passages:")
    print("\n======\n".join(retrieved_passages[:3]))


def main():
    res = debug_hp_token_len()
    if __file__.startswith('/Users/joey/'):
        download_path = '/Users/joey/Downloads/hp_qwen2.tar.bz2'
        extract_path = '/Users/joey/Downloads/'
    else:
        download_path = '/content/hp_qwen2.tar.bz2'
        extract_path = '/content/'
    download_chunks_url(
        url='https://huggingface.co/datasets/TommyChien/MemoRAG-data/resolve/main/hp_qwen2.tar.bz2',
        download_path=download_path,
        extract_path=extract_path
    )
    pipe = get_pipe()
    load_weights_from_disk(pipe)
    run_pipe(pipe)
    qa_hp_files(pipe, res)
