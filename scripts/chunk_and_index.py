import os, json, pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

IN_PATH = Path("data/processed/ingested.jsonl")
PROCESSED_DIR = Path("data/processed")
ART_DIR = Path("artifacts")
ART_DIR.mkdir(parents=True, exist_ok=True)

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    tokens = text.split(" ")
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+size]
        chunks.append(" ".join(chunk))
        i += size - overlap
    return chunks

def load_ingested() -> List[Dict]:
    rows = []
    with IN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def main():
    docs = load_ingested()
    model = SentenceTransformer(EMB_MODEL)

    texts, metadatas = [], []
    for d in docs:
        for ch in chunk_text(d["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            if len(ch) < 200: 
                continue
            texts.append(ch)
            metadatas.append({"doc_id": d["id"], "source": d["source"]})

    # Embeddings
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    np.save(ART_DIR / "embeddings.npy", embeddings)
    with open(ART_DIR / "docstore.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metadatas": metadatas}, f)
    faiss.write_index(index, str(ART_DIR / "index.faiss"))

    print(f"Built index with {len(texts)} chunks")

if __name__ == "__main__":
    main()
