import numpy as np
from src.embeddings import chunk_text, embed_text
from src.vectorstore import save_vectorstore, load_vectorstore, search


def retrieve(query, path="data/vectorstore.json", k=1):
    question = np.array(embed_text(query),dtype=np.float32)
    embeddings, chunks = load_vectorstore(path)
    answer = search(question, embeddings, chunks, k=k)

    return answer