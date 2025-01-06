#!/bin/bash
venv_name="auto_file_sorter_venv"

rm -rf "$venv_name"

# Prefer Python 3.11 for better compatibility
if command -v python3.11 &>/dev/null; then
    python_cmd="python3.11"
elif command -v python3.10 &>/dev/null; then
    python_cmd="python3.10"
else
    echo "Python 3.10 or 3.11 is required but not found. Please install it and try again."
    exit 1
fi

echo "Creating virtual environment using $python_cmd"

# Create virtual environment
$python_cmd -m venv "$venv_name"
source "./$venv_name/bin/activate"
echo "Installing dependencies..."

# TODO: Check if there is a requirements.txt file and if it exists, install the dependencies from it.
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt file found. Installing dependencies from pip..."
fi

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install CUDA toolkit if needed
if [[ "$(uname)" == "Linux" ]]; then
    sudo apt-get install -y nvidia-cuda-toolkit
fi

# Install transformers and its dependencies
pip install transformers accelerate

# Install memorag dependencies first
pip install ninja
pip install triton

# Try installing memorag with specific version
pip install "memorag>=0.1.5,<0.2.0"

# Install other dependencies
pip install deepdiff decouple pytz python-dateutil
pip install requests python-dotenv pyngrok watchdog
pip install pathlib
pip install pydub audioread Pillow pycryptodome
pip install "tiktoken<0.5.0" openai anthropic
pip install "langchain>=0.0.350" langchain_anthropic langchain_openai langchain_core langchain_community
pip install PyPDF2 nltk mutagen python-docx python-pptx openpyxl
pip install fpdf bs4
pip install google-auth google_auth_oauthlib google-api-python-client
pip install fastapi uvicorn websockets

echo "Virtual environment created: $VIRTUAL_ENV"

# Generate requirements file
pip freeze >requirements.txt
echo "requirements.txt file created"

# Verify installations
echo "Verifying key installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import memorag; print(f'Memorag version: {memorag.__version__}')"
