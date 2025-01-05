from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from .document_processor import DocumentProcessor


class DocumentManager:
    """Manages document storage, retrieval and organization"""

    def __init__(
        self,
        base_dir: str = "memrag/data",
        hf_token: Optional[str] = None
    ):
        self.base_dir = Path(base_dir)
        self.docs_dir = self.base_dir / "documents"
        self.index_path = self.base_dir / "document_index.json"

        # Create directories
        self.docs_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processor
        self.processor = DocumentProcessor(
            cache_dir=str(self.base_dir / "cache"),
            hf_token=hf_token
        )

        # Load document index
        self.doc_index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """Load document index from disk"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_index(self):
        """Save document index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.doc_index, f, indent=2)

    def add_document(self, content: str, filename: str) -> Dict[str, Any]:
        """Add a new document to the system"""
        try:
            # Process document
            doc_info = self.processor.process_document(content, filename)

            # Save to index
            self.doc_index[filename] = doc_info
            self._save_index()

            return doc_info

        except Exception as e:
            logging.error(f"Error adding document {filename}: {e}")
            raise

    def search_documents(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents using MemoRAG"""
        try:
            # Generate search clues
            clues = self.processor.pipe.mem_model.rewrite(query).split("\n")
            clues = [q for q in clues if len(q.split()) > 3]

            # Get relevant documents
            results = []
            for doc_id, doc_info in self.doc_index.items():
                relevance = self.processor.pipe(
                    context=doc_info["summary"],
                    query=query,
                    task_type="qa",
                    max_new_tokens=128
                )
                results.append((doc_id, doc_info, relevance))

            # Sort by relevance
            results.sort(key=lambda x: len(x[2]), reverse=True)

            return [
                {
                    "doc_id": doc_id,
                    "summary": doc_info["summary"],
                    "tags": doc_info["tags"],
                    "relevance": relevance
                }
                for doc_id, doc_info, relevance in results[:max_results]
            ]

        except Exception as e:
            logging.error(f"Error searching documents: {e}")
            raise
