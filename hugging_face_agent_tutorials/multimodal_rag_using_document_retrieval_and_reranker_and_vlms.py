# # Multimodal RAG with ColQwen2, Reranker, and Quantized VLMs on Consumer GPUs
#
# _Authored by: [Sergio Paniego](https://github.com/sergiopaniego)_
#

# In this notebook, we demonstrate how to build a **Multimodal Retrieval-Augmented Generation (RAG)** system by integrating [**ColQwen2**](https://huggingface.co/vidore/colqwen2-v1.0) for document retrieval, [**MonoQwen2-VL-v0.1**](https://huggingface.co/lightonai/MonoQwen2-VL-v0.1) for reranking, and [**Qwen2-VL**](https://qwenlm.github.io/blog/qwen2-vl/) as the vision language model (VLM). Together, these models form a powerful RAG system that enhances query responses by seamlessly combining text-based documents and visual data. Notably, this notebook is optimized for use on a single consumer GPU, thanks to the integration of a quantized VLM.
#
# Instead of relying on a complex OCR-based document processing pipeline, we leverage a **Document Retrieval Model** to efficiently retrieve the most relevant documents based on a user’s query, making the system more scalable and efficient.
#
# This notebook builds on the concepts introduced in our previous guide, [**Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs)**](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms). If you haven't reviewed that notebook yet, we recommend doing so before proceeding with this one.
#
# Tested on an L4 GPU.

# Image: multimodal_rag_using_document_retrieval_and_reranker_and_vlms_2 (1).png

# This diagram is inspired by [Aymeric Roucher's](https://huggingface.co/m-ric) work in the [Advanced RAG](https://huggingface.co/learn/cookbook/advanced_rag) or [RAG Evaluation](https://huggingface.co/learn/cookbook/rag_evaluation) recipes.

# ## 1. Install dependencies
# Let’s kick off by installing the essential libraries for our project! 🚀
#

import subprocess
subprocess.run(['pip', 'install', '-U', '-q', 'byaldi', 'pdf2image', 'qwen-vl-utils', 'transformers', 'bitsandbytes', 'peft'], check=True)
# Tested with byaldi==0.0.7, pdf2image==1.17.0, qwen-vl-utils==0.0.8, transformers==4.46.3
import subprocess
subprocess.run(['pip', 'install', '-U', '-q', 'rerankers[monovlm]'], check=True)

# ## 2. Load Dataset 📁
#
# For this notebook, we will use charts and maps from [Our World in Data](https://ourworldindata.org/), a valuable resource offering open access to a wide range of data and visualizations. Specifically, we will focus on the [life expectancy data](https://ourworldindata.org/life-expectancy).
#
# To facilitate easy access, we've curated a small subset of this data in the following [dataset](https://huggingface.co/datasets/sergiopaniego/ourworldindata_example).
#
# While we have selected a few examples from this source for demonstration purposes, in a real-world scenario, you could work with a much larger collection of visual data to further enhance your model's capabilities.
#
# **Citation:**
#
# ```
# Saloni Dattani, Lucas Rodés-Guirao, Hannah Ritchie, Esteban Ortiz-Ospina and Max Roser (2023) - “Life Expectancy” Published online at OurWorldinData.org. Retrieved from: 'https://ourworldindata.org/life-expectancy' [Online Resource]
# ```

from datasets import load_dataset

dataset = load_dataset("sergiopaniego/ourworldindata_example", split='train')
# After downloading the visual data, we will save it locally to enable the RAG (Retrieval-Augmented Generation) system to index the files later. This step is crucial, as it allows the document retrieval model (ColQwen2) to efficiently process and manipulate the visual content. Additionally, we reduce the image size to **448x448** to further minimize memory consumption and ensure faster processing, which is important for optimizing performance in large-scale operations.
#

import os
from PIL import Image

def save_images_to_local(dataset, output_folder="data/"):
    os.makedirs(output_folder, exist_ok=True)

    for image_id, image_data in enumerate(dataset):
        image = image_data['image']

        if isinstance(image, str):
            image = Image.open(image)

        image = image.resize((448, 448))

        output_path = os.path.join(output_folder, f"image_{image_id}.png")

        image.save(output_path, format='PNG')

        print(f"Image saved in: {output_path}")

save_images_to_local(dataset)
# Now, let's load the images to explore the data and get an overview of the visual content.
import os
from PIL import Image

def load_png_images(image_folder):
    png_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    all_images = {}

    for image_id, png_file in enumerate(png_files):
        image_path = os.path.join(image_folder, png_file)
        image = Image.open(image_path)
        all_images[image_id] = image

    return all_images

all_images = load_png_images("/content/data/")
# Let’s visualize a few samples to get an understanding of how the data is structured! This will help us grasp the format and layout of the content we’ll be working with. 👀
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 5, figsize=(20, 15))

for i, ax in enumerate(axes.flat):
    img = all_images[i]
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()
# ## 3. Initialize the ColQwen2 Multimodal Document Retrieval Model 🤖

# Now that our dataset is ready, we will initialize the Document Retrieval Model, which will be responsible for extracting relevant information from the raw images and delivering the appropriate documents based on our queries.
#
# Using this model, we can greatly enhance our system’s conversational capabilities.
#
# For this task, we will use **[Byaldi](https://github.com/AnswerDotAI/byaldi)**. The developers describe the library as follows: _"Byaldi is RAGatouille's mini sister project. It is a simple wrapper around the ColPali repository to make it easy to use late-interaction multi-modal models such as ColPALI with a familiar API."_
#
# In this project, we will focus specifically on **ColQwen2**.
#
# Image: ColPali architecture
#
# Additionally, you can explore **[ViDore (The Visual Document Retrieval Benchmark)](https://huggingface.co/spaces/vidore/vidore-leaderboard)** to see the top-performing retrievers in action.
#

# First, we will load the model from the checkpoint.

from byaldi import RAGMultiModalModel

docs_retrieval_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")
# Next, we can index our documents directly using the document retrieval model by specifying the folder where the images are stored. This will enable the model to process and organize the documents for efficient retrieval based on our queries.

docs_retrieval_model.index(
    input_path="data/",
    index_name="image_index",
    store_collection_with_index=False,
    overwrite=True
)
# ## 4. Retrieving Documents with the Document Retrieval Model and Re-ranking with the Reranker 🤔
#
# Now that the document retrieval model is initialized, we can test its capabilities by submitting a user query and reviewing the relevant documents it retrieves.
#
# The model will return the results ranked by their relevance to the query. Next, we’ll use the reranker to further enhance the retrieval pipeline.
#
# Let’s give it a try!

text_query = 'How does the life expectancy change over time in France and South Africa?'

results = docs_retrieval_model.search(text_query, k=3)
results
# Now, let’s examine the specific documents (images) the model has retrieved. This will give us insight into the visual content that corresponds to our query and help us understand how the model selects relevant information.

def get_grouped_images(results, all_images):
    grouped_images = []

    for result in results:
        doc_id = result['doc_id']
        page_num = result['page_num']
        grouped_images.append(all_images[doc_id])
    return grouped_images

grouped_images = get_grouped_images(results, all_images)
# Let’s take a closer look at the retrieved documents to better understand the information they contain. This examination will help us assess the relevance and quality of the content in relation to our query.

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    img = grouped_images[i]
    ax.imshow(img)
    ax.axis('off')

plt.tight_layout()
plt.show()
# As you can see, the retrieved documents are relevant to the query, as they contain related data.
#
# Now, let's initialize our reranker model. For this, we will use the [rerankers](https://github.com/AnswerDotAI/rerankers) module.
from rerankers import Reranker

ranker = Reranker("monovlm", device='cuda')
# The reranker requires the images to be in base64 format, so let's start by converting the images before proceeding with the reranking.

import base64
from io import BytesIO

def images_to_base64(images):
    base64_images = []
    for img in images:
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)

        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        base64_images.append(img_base64)

    return base64_images

base64_list = images_to_base64(grouped_images)
# Once again, we pass the `text_query` and the list of images to the reranker so it can enhance the retrieved context. This time, instead of using the 3 previously retrieved documents, we will return only 1. If you look at the results, you’ll notice that the model assigns the majority of the score to just one image, improving the ranking from the previous iteration.

results = ranker.rank(text_query, base64_list)
def process_ranker_results(results, grouped_images, top_k=3, log=False):
    new_grouped_images = []
    for i, doc in enumerate(results.top_k(top_k)):
        if log:
          print(f"Rank {i}:")
          print("Document ID:", doc.doc_id)
          print("Document Score:", doc.score)
          print("Document Base64:", doc.base64[:30] + '...')
          print("Document Path:", doc.image_path)
        new_grouped_images.append(grouped_images[doc.doc_id])
    return new_grouped_images
new_grouped_images = process_ranker_results(results, grouped_images, top_k=1, log=True)
# After that, we’re ready to load the VLM and generate the response to the user query!
# ## 5. Initialize the Visual Language Model for Question Answering 🙋
#
# Next, we’ll initialize the Visual Language Model (VLM) for question answering. For this, we’ll be using **[Qwen2_VL](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl)**.
#
# Image: Qwen2_VL architecture
#
# Stay up to date with the latest advancements in Open VLM by checking the leaderboard [here](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard).
#
# To begin, we’ll load the model from the pretrained checkpoint and move it to the GPU for optimal performance. You can find the model [here](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct).
#
# In this notebook, we are using a **quantized version** of the model to optimize memory usage and processing speed, which is especially important when running on a consumer GPU. By utilizing a quantized version, we reduce the model’s memory footprint and improve its efficiency while maintaining performance for the task at hand.
#

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch

# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
vl_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)
vl_model.eval()
# Next, we will initialize the Visual Language Model (VLM) processor. In this step, we specify the minimum and maximum pixel sizes to optimize how images fit into the GPU memory. The larger the pixel size, the more memory it will consume, so it’s important to find a balance that ensures optimal performance without overloading the GPU.
#
# For more details on how to optimize image resolution for performance, you can refer to the [documentation here](https://huggingface.co/docs/transformers/main/en/model_doc/qwen2_vl#image-resolution-for-performance-boost).

min_pixels = 224*224
max_pixels = 448*448
vl_model_processor = Qwen2VLProcessor.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
# ## 6. Assembling the VLM Model and Testing the System 🔧
#
# With all components loaded, we are ready to assemble the system for testing. First, we’ll set up the chat structure by providing the system with the retrieved image(s) and the user’s query. This step is highly customizable, offering flexibility to adjust the interaction according to your needs and enabling experimentation with different inputs and outputs.
chat_template = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": new_grouped_images[0],
            },
            {
                "type": "text",
                "text": text_query
            },
        ],
    }
]
# Now, let’s apply this chat template to set up the system for interacting with the model.

text = vl_model_processor.apply_chat_template(
    chat_template, tokenize=False, add_generation_prompt=True
)
# Next, we will process the inputs to ensure they are properly formatted and ready for use with the Visual Language Model (VLM). This step is crucial for enabling the model to generate accurate responses based on the provided data.
image_inputs, _ = process_vision_info(chat_template)
inputs = vl_model_processor(
    text=[text],
    images=image_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
# We are now ready to generate the answer! Let’s see how the system uses the processed inputs to provide a response based on the user query and the retrieved images.

generated_ids = vl_model.generate(**inputs, max_new_tokens=500)
# Once the model generates the output, we postprocess it to generate the final answer.
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = vl_model_processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
# ## 7. Assembling It All! 🧑‍🏭️
#
# Now, let’s create a method that encompasses the entire pipeline, allowing us to easily reuse it in future applications.

def answer_with_multimodal_rag(vl_model, docs_retrieval_model, vl_model_processor, grouped_images, text_query, retrival_top_k, reranker_top_k, max_new_tokens):
    results = docs_retrieval_model.search(text_query, k=retrival_top_k)
    grouped_images = get_grouped_images(results, all_images)

    base64_list = images_to_base64(grouped_images)
    results = ranker.rank(text_query, base64_list)
    grouped_images = process_ranker_results(results, grouped_images, top_k=reranker_top_k)

    chat_template = [
    {
      "role": "user",
      "content": [
          {"type": "image", "image": image} for image in grouped_images
            ] + [
          {"type": "text", "text": text_query}
        ],
      }
    ]

    # Prepare the inputs
    text = vl_model_processor.apply_chat_template(chat_template, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(chat_template)
    inputs = vl_model_processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate text from the vl_model
    generated_ids = vl_model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated text
    output_text = vl_model_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text
# Let’s take a look at how the complete RAG system operates!

output_text = answer_with_multimodal_rag(
    vl_model=vl_model,
    docs_retrieval_model=docs_retrieval_model,
    vl_model_processor=vl_model_processor,
    grouped_images=grouped_images,
    text_query='What is the overall trend in life expectancy across different countries and regions?',
    retrival_top_k=3,
    reranker_top_k=1,
    max_new_tokens=500
)
print(output_text[0])
import torch
torch.cuda.empty_cache()
torch.cuda.synchronize()
print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
# 🏆 We now have a fully operational RAG pipeline that integrates both a Document Retrieval Model and a Visual Language Model, optimized to run on a single consumer GPU! This powerful combination allows us to generate insightful responses based on user queries and relevant documents.
#
# Additionally, we’ve implemented a reranking step to further refine the document retrieval process, improving the relevance of the results and enhancing the overall performance of the system.
#

# ## 8. Continuing the Journey 🧑‍🎓️
#
# If you're eager to continue exploring, be sure to check out the results and insights in the conclusion of our previous guide, [**Multimodal Retrieval-Augmented Generation (RAG) with Document Retrieval (ColPali) and Vision Language Models (VLMs)**](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms). It's a great next step to deepen your understanding of multimodal RAG systems!
