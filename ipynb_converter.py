import json
import sys
import re
from pathlib import Path

# ### Usage
# # Convert a single notebook
# python ipynb_converter.py path/to/notebook.ipynb

# # Convert multiple notebooks
# python ipynb_converter.py notebook1.ipynb notebook2.ipynb

# # Convert all notebooks in current directory
# python ipynb_converter.py *.ipynb


def convert_shell_command(code_line: str) -> str:
    """Convert a shell command line to a Python subprocess call.

    Args:
        code_line: The shell command line starting with ! or !!

    Returns:
        Python code that executes the shell command using subprocess
    """
    # Remove the leading ! or !!
    cmd = code_line.lstrip('!')

    # Split the command into parts and escape quotes
    cmd_parts = cmd.strip().split()
    cmd_parts = [f"'{part}'" for part in cmd_parts]

    return f"subprocess.run([{', '.join(cmd_parts)}], check=True)\n"


def process_code_cell(code_lines: list) -> list:
    """Process a code cell, converting shell commands if present.

    Args:
        code_lines: List of code lines from the cell

    Returns:
        Processed code lines
    """
    processed_lines = []

    # Check if entire cell is shell script (starts with !!)
    if code_lines and code_lines[0].startswith('!!'):
        # Join all lines into a single shell script
        shell_script = ''.join(line.lstrip('!') for line in code_lines)
        processed_lines.extend([
            "import subprocess\n",
            f"subprocess.run('''{shell_script}''', shell=True, check=True)\n"
        ])
        return processed_lines

    # Process individual lines
    for line in code_lines:
        if line.lstrip().startswith('!'):
            # Add subprocess import if not already added
            if not any('import subprocess' in l for l in processed_lines):
                processed_lines.insert(0, "import subprocess\n")
            processed_lines.append(convert_shell_command(line))
        else:
            processed_lines.append(line)

    return processed_lines


def process_markdown_cell(markdown_lines: list) -> list:
    """Process markdown cell content, handling images specially.

    Args:
        markdown_lines: List of markdown lines

    Returns:
        List of processed lines as Python comments
    """
    commented_lines = []

    for line in markdown_lines:
        # Skip empty lines
        if not line.strip():
            commented_lines.append("#\n")
            continue

        # Check if line contains an image
        img_match = re.search(r'!\[(.*?)\]', line)
        if img_match and len(line) > 100:
            # Only include alt text if present
            alt_text = img_match.group(1)
            if alt_text:
                commented_lines.append(f"# Image: {alt_text}\n")
            continue

        # Regular markdown line
        commented_lines.append(f"# {line}")

    return commented_lines


def convert_notebook_to_python(notebook_path: str) -> str:
    """Convert a Jupyter notebook to a Python script.

    Args:
        notebook_path: Path to the .ipynb file

    Returns:
        String containing the converted Python script content
    """
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    python_code = []

    # Process each cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'markdown':
            # Convert markdown to Python comments
            commented_lines = process_markdown_cell(cell['source'])
            python_code.extend(commented_lines)
            python_code.append("\n")

        elif cell['cell_type'] == 'code':
            # Process code cell, handling shell commands
            code_lines = cell['source']
            if code_lines:  # Only add if there's actual code
                processed_lines = process_code_cell(code_lines)
                python_code.extend(processed_lines)
                python_code.append("\n")

    return "".join(python_code)


def process_notebook(notebook_path: str) -> None:
    """Process a single notebook file and save the converted Python script.

    Args:
        notebook_path: Path to the .ipynb file
    """
    input_path = Path(notebook_path)

    # Skip if not a notebook file
    if input_path.suffix != '.ipynb':
        print(f"Skipping {input_path} - not a notebook file")
        return

    # Create output path with .py extension
    output_path = input_path.with_suffix('.py')

    try:
        # Convert the notebook
        python_content = convert_notebook_to_python(str(input_path))

        # Write the Python file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(python_content)

        print(f"Successfully converted {input_path} to {output_path}")

    except Exception as e:
        print(f"Error converting {input_path}: {str(e)}")


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print(
            "Usage: python ipynb_converter.py <notebook_path> [notebook_path2 ...]")
        sys.exit(1)

    # Process each notebook file provided as argument
    for notebook_path in sys.argv[1:]:
        process_notebook(notebook_path)


if __name__ == "__main__":
    main()
