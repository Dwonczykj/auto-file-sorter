#!/bin/bash
venv_name="memrag_venv"

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

# Upgrade pip and install build tools
pip install --upgrade pip setuptools wheel

# Install PyTorch and CUDA dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ninja triton

# Install memorag and its core dependencies
pip install memorag==0.1.3
pip install flash-attn --no-deps
pip install -U bitsandbytes
pip install faiss-cpu # Use faiss-gpu if GPU is available

# Install document processing dependencies
pip install python-docx PyPDF2
pip install tiktoken
pip install transformers accelerate

# Install utility dependencies
pip install python-dotenv
pip install tqdm
pip install requests

echo "Virtual environment created: $VIRTUAL_ENV"

# Create necessary directories
mkdir -p memrag/{core,utils,models,data/{cache,documents}}

# Create __init__.py files
touch memrag/__init__.py
touch memrag/core/__init__.py
touch memrag/utils/__init__.py
touch memrag/models/__init__.py

# Generate requirements file
pip freeze >requirements.txt
echo "requirements.txt file created"

# Verify installations
echo "Verifying key installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import memorag; print(f'Memorag version: {memorag.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Setup complete! Use 'source memrag_venv/bin/activate' to activate the virtual environment."
