from typing import List, Dict, Any, Optional
from pathlib import Path
import tiktoken
from memorag import MemoRAG
import logging


class DocumentProcessor:
    """Handles document processing and memory management with MemoRAG"""

    def __init__(
        self,
        cache_dir: str = "memrag/data/cache",
        hf_token: Optional[str] = None,
        beacon_ratio: int = 4
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize MemoRAG pipeline
        self.pipe = MemoRAG(
            mem_model_name_or_path="TommyChien/memorag-mistral-7b-inst",
            ret_model_name_or_path="BAAI/bge-m3",
            gen_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir=str(self.cache_dir),
            access_token=hf_token,
            beacon_ratio=beacon_ratio,
            load_in_4bit=True
        )

        self.encoding = tiktoken.get_encoding("cl100k_base")

    def process_document(self, content: str, doc_id: str) -> Dict[str, Any]:
        """Process a document and store it in memory"""
        try:
            # Count tokens
            token_count = len(self.encoding.encode(content))
            logging.info(f"Processing document {
                         doc_id} with {token_count} tokens")

            # Memorize content
            save_path = self.cache_dir / f"doc_{doc_id}"
            self.pipe.memorize(
                content,
                save_dir=str(save_path),
                print_stats=True
            )

            # Generate summary and tags
            summary = self.pipe(
                context=content,
                query="Please provide a concise summary of this document.",
                task_type="memorag",
                max_new_tokens=256
            )

            tags = self.pipe(
                context=content,
                query="What are the main topics and keywords in this document? Return as comma-separated list.",
                task_type="memorag",
                max_new_tokens=128
            )

            return {
                "doc_id": doc_id,
                "token_count": token_count,
                "summary": summary[0] if isinstance(summary, list) else summary,
                "tags": [t.strip() for t in tags[0].split(",")],
                "cache_path": str(save_path)
            }

        except Exception as e:
            logging.error(f"Error processing document {doc_id}: {e}")
            raise
