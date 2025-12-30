from pathlib import Path
from pypdf import PdfReader 
from src.embeddings import chunk_text, model
from src.vectorstore import save_vectorstore

DOCS_PATH = Path("documents")
VECTORSTORE_PATH = "data/vectorstore.json"

def build_vectorstore():
    extensions = {'.txt', '.pdf'}
    all_chunks = []
    for file_path in DOCS_PATH.rglob("*"):
        if file_path.suffix.lower() not in extensions:
            continue
        try:
            if file_path.suffix.lower() == '.txt':
                text = file_path.read_text(encoding='utf-8')
            elif file_path.suffix.lower() == '.pdf':
                reader = PdfReader(file_path)
                text = "".join([page.extract_text() for page in reader.pages])
            raw_chunks = chunk_text(text,350)
          
            for i, c in enumerate(raw_chunks):
                all_chunks.append({
                    "text": c,
                    "chunk_id": i,
                    "doc_name": file_path.name
                })
        except Exception as e:
            print(f"Could not read {file_path.name}: {e}")        
    embeddings = model.encode([c["text"] for c in all_chunks])
    save_vectorstore(embeddings, all_chunks, VECTORSTORE_PATH)
    print(f"Saved {len(all_chunks)} chunks to vectorstore.")
    
    for file_path in DOCS_PATH.rglob("*"):
        print("Processed:", file_path.name)

if __name__ == "__main__":
    build_vectorstore()


