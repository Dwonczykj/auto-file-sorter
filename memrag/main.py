import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
from core.document_manager import DocumentManager
from utils.file_handler import FileHandler


class DocumentManagementSystem:
    """Main interface for the document management system"""

    def __init__(
        self,
        base_dir: str = "memrag/data",
        hf_token: str = None
    ):
        self.manager = DocumentManager(base_dir, hf_token)
        self.file_handler = FileHandler()

    def add_file(self, file_path: str) -> Dict[str, Any]:
        """Add a file to the system"""
        try:
            content = self.file_handler.read_file(file_path)
            filename = Path(file_path).name
            return self.manager.add_document(content, filename)

        except Exception as e:
            logging.error(f"Error adding file {file_path}: {e}")
            raise

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents"""
        return self.manager.search_documents(query, max_results)

    def get_document_info(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific document"""
        return self.manager.doc_index.get(doc_id)


def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)

    # Get Hugging Face token from environment
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Initialize system
    dms = DocumentManagementSystem(hf_token=hf_token)

    # Example usage
    try:
        # Add a document
        doc_info = dms.add_file("path/to/document.pdf")
        print(f"Added document: {doc_info}")

        # Search documents
        results = dms.search(
            "What are the main topics about machine learning?")
        print(f"Search results: {results}")

    except Exception as e:
        logging.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()
