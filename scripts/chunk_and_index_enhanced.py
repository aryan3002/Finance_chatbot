"""Enhanced chunking script to handle various financial document types."""
import json
from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(base_dir: Path) -> List[Dict]:
    """Load all processed documents."""
    documents = []
    
    # Load regulatory documents
    with open("data/processed/ingested.jsonl", "r") as f:
        for line in f:
            documents.append(json.loads(line))
    
    # Load financial documents
    finance_dir = Path("data/processed/finance")
    if finance_dir.exists():
        for file_path in finance_dir.glob("*_processed.json"):
            with open(file_path, 'r') as f:
                doc = json.loads(f.read())
                # Convert to chunks
                chunks = chunk_financial_document(doc)
                documents.extend(chunks)
    
    return documents

def chunk_financial_document(doc: Dict) -> List[Dict]:
    """Convert a processed financial document into chunks."""
    chunks = []
    
    # Add chunks from raw text
    text = doc["raw_text"]
    chunk_size = 1000
    overlap = 200
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if len(chunk_text) < 50:  # Skip very small chunks
            continue
            
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source": doc["metadata"]["source"],
                "category": doc["metadata"]["category"],
                "chunk_id": len(chunks)
            }
        })
    
    # Add specific concept chunks
    for category, concept_list in doc["concepts"].items():
        for concept in concept_list:
            chunks.append({
                "text": concept,
                "metadata": {
                    "source": doc["metadata"]["source"],
                    "category": f"{doc['metadata']['category']}_concepts",
                    "concept_type": category,
                    "chunk_id": len(chunks)
                }
            })
    
    return chunks

def create_index(documents: List[Dict], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[faiss.Index, Dict]:
    """Create FAISS index from documents."""
    # Load the encoder
    encoder = SentenceTransformer(model_name)
    
    # Prepare texts and metadata
    texts = []
    metadatas = []
    
    for doc in documents:
        texts.append(doc["text"])
        metadatas.append(doc["metadata"])
    
    # Create embeddings
    embeddings = encoder.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    
    # Create docstore
    docstore = {
        "texts": texts,
        "metadatas": metadatas
    }
    
    return index, docstore

def main():
    # Load all documents
    documents = load_documents(Path("data/processed"))
    logger.info(f"Loaded {len(documents)} documents")
    
    # Create index and docstore
    index, docstore = create_index(documents)
    
    # Save artifacts
    artifact_dir = Path("artifacts")
    artifact_dir.mkdir(exist_ok=True)
    
    faiss.write_index(index, str(artifact_dir / "index.faiss"))
    with open(artifact_dir / "docstore.pkl", "wb") as f:
        pickle.dump(docstore, f)
    
    logger.info("Successfully created and saved index and docstore")

if __name__ == "__main__":
    main()
