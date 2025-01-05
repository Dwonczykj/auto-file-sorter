import os
import logging
from pathlib import Path
from main import DocumentManagementSystem


def test_system():
    """Test the document management system functionality"""
    logging.basicConfig(level=logging.INFO)

    # Create test document
    test_doc = """
    Machine Learning Overview
    
    Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
    
    Key concepts include:
    1. Supervised Learning
    2. Unsupervised Learning
    3. Deep Learning
    4. Neural Networks
    
    Applications range from image recognition to natural language processing.
    """

    test_file = Path("memrag/data/test_doc.txt")
    test_file.parent.mkdir(parents=True, exist_ok=True)

    with open(test_file, "w") as f:
        f.write(test_doc)

    try:
        # Initialize system
        dms = DocumentManagementSystem(
            hf_token=os.getenv("HUGGINGFACE_TOKEN")
        )

        # Test adding document
        print("\nTesting document addition...")
        doc_info = dms.add_file(str(test_file))
        print(f"Document added successfully: {doc_info}")

        # Test search functionality
        print("\nTesting search functionality...")
        queries = [
            "What are the main topics?",
            "Explain deep learning concepts",
            "What are the applications?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            results = dms.search(query)
            print(f"Results: {results}")

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"Error during testing: {e}")
        raise
    finally:
        # Cleanup
        if test_file.exists():
            test_file.unlink()


if __name__ == "__main__":
    test_system()
