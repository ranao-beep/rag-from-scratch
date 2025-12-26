import json
import numpy as np
from pathlib import Path

def save_vectorstore(embeddings, chunks, path="data/vectorstore.json"):
    """
    embeddings: np.ndarray of shape (N, dim)
    chunks: list (strings or dicts with metadata)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "embeddings": embeddings.tolist(),  # numpy → list
        "chunks": chunks
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_vectorstore(path="data/vectorstore.json"):
    """
    Returns:
        embeddings: np.ndarray
        chunks: list
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.array(data["embeddings"], dtype=np.float32)
    chunks = data["chunks"]

    return embeddings, chunks

def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a, b)

def search(query_embedding, embeddings, chunks, k):
    similarities = []
    for idx,embed in enumerate(embeddings):
        simScore = cosine_similarity(query_embedding,embed)
        similarities.append((idx,simScore))
    similarities.sort(key=lambda x: x[1],reverse=True)

    top_k = similarities[:k]
    # for s in similarities:
    #     if len(top_k) != k:
    #         top_k.append(s)
    # return top_k
    
    results = []
    for idx, simScore in top_k:
        results.append({
            "chunk":chunks[idx],
            "score":simScore
        })
    return results