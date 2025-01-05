from typing import List, Dict, Any, Optional
from pathlib import Path
import mimetypes
import logging
from PyPDF2 import PdfReader
from docx import Document
import csv


class FileHandler:
    """Handles reading different file formats"""

    @staticmethod
    def read_file(file_path: str) -> str:
        """Read content from various file types"""
        path = Path(file_path)
        mime_type = mimetypes.guess_type(file_path)[0]

        try:
            if mime_type == 'text/plain':
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()

            elif mime_type == 'application/pdf':
                reader = PdfReader(path)
                return ' '.join(page.extract_text() for page in reader.pages)

            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                doc = Document(path)
                return ' '.join(paragraph.text for paragraph in doc.paragraphs)

            elif mime_type == 'text/csv':
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    return '\n'.join([','.join(row) for row in reader])

            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            raise
