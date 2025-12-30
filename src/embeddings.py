import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def chunk_text(text:str, max_chunk_size: int=500) -> list[str]:

    text = clean_text(text)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    current = []
    chunks = []
    current_len = 0
    for s in sentences:
        s_len = len(s)
        if s_len + current_len <=max_chunk_size:
            current.append(s)
            current_len += s_len 
        else:
            if current:
                chunks.append(" ".join(current))
            s_overlap = current[-1:]
            current = []
            current.extend(s_overlap)
            current.append(s)
            current_len = s_len 
    if current:
        chunks.append(" ".join(current))

    return chunks    

def embed_text(text: str) -> list[float]:
    """
    ffffd
    Returns a single embedding vector for the given text.
    """
    text = clean_text(text)
    embedding = model.encode(text)
    return embedding

