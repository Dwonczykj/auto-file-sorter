#!/bin/bash
venv_name="auto_file_sorter_venv"

rm -rf "$venv_name"

if command -v python3.14 &>/dev/null; then
    python_cmd="python3.14"
elif command -v python3.13 &>/dev/null; then
    python_cmd="python3.13"
elif command -v python3.12 &>/dev/null; then
    python_cmd="python3.12"
elif command -v python3.11 &>/dev/null; then
    python_cmd="python3.11"
elif command -v python3.10 &>/dev/null; then
    python_cmd="python3.10"
else
    echo "Python 3.10 or later is required but not found. Please install it and try again."
    exit 1
fi

echo "Creating virtual environment using $python_cmd"

# Create a new virtual environment using Python 3.10 or later
$python_cmd -m venv "$venv_name"
# shellcheck disable=SC1090
source "./$venv_name/bin/activate" # On Windows use `whatsapp_transcriber_venv\Scripts\activate`
echo "Installing dependencies..."

# shellcheck disable=SC1090
source "./$venv_name/bin/activate" && pip freeze >requirements.txt # On Windows use `whatsapp_transcriber_venv\Scripts\activate`
pip install deepdiff decouple pytz python-dateutil
pip install requests python-dotenv pyngrok watchdog
pip install torch transformers pathlib
pip install pydub audioread Pillow pycryptodome
pip install tiktoken openai anthropic langchain langchain_anthropic langchain_openai langchain_core langchain_community
pip install PyPDF2 nltk mutagen python-docx PyPDF2 python-pptx openpyxl
pip install fpdf bs4
pip install google-auth google_auth_oauthlib google-api-python-client

echo "Virtual environment created: $VIRTUAL_ENV"

# shellcheck disable=SC2086
echo "# Virtual Environment: $(basename $VIRTUAL_ENV)" >requirements.txt && pip freeze >>requirements.txt

echo "requirements.txt file created"
# conda create --name "$venv_name" python=3.11 -c conda-forge -y requests python-dotenv openai pyngrok pydub audioread tiktoken pycryptodome Pillow anthropic PyPDF2 watchdog nltk mutagen python-docx PyPDF2 python-pptx openpyxl
# conda activate "$venv_name"
