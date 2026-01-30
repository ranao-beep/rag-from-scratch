# RAG from Scratch

This repository contains a small **Retrieval-Augmented Generation (RAG)** project built from scratch as a learning exercise.

The goal of this project is not to build a production-ready system, but to understand how the core components of a RAG pipeline work internally by implementing them step by step and debugging them manually.

---

## Project Description

The system allows asking natural language questions over local documents (PDF or TXT).  
Relevant text segments are retrieved using embedding-based similarity search, and answers are generated only when supporting source text is found.

Instead of using high-level RAG frameworks, the project focuses on implementing the main ideas in a simple and transparent way.

---

## How It Works

The pipeline follows these steps:

- **Document ingestion**  
  PDF files are loaded using `pypdf` and text is extracted page by page.

- **Text chunking**  
  Extracted text is split into small chunks (sentence-level or fixed-size).  
  Each chunk is stored with a unique ID and document reference.

- **Embeddings**  
  Each chunk is converted into a vector representation.  
  Embeddings and metadata are stored locally.

- **Vector retrieval**  
  User queries are embedded and compared to stored embeddings using similarity search.  
  The most relevant chunks are selected.

- **Inference and output**  
  Retrieved chunks are returned as sources.  
  If no relevant chunk is found, the system explicitly reports that no grounded answer is available.

---

## What This Project Focuses On

- Understanding the internal structure of a RAG system
- Learning how chunking choices affect retrieval results
- Debugging retrieval failures instead of hiding them
- Inspecting sources instead of relying blindly on model output

---

## Known Limitations

This project is intentionally limited in scope:

- Simple chunking strategy (no semantic or adaptive chunking)
- No reranking or hybrid search
- Retrieval quality depends strongly on chunk size
- Not optimized for large document collections
- Not intended for direct production use

These limitations helped reveal common RAG failure cases during testing, such as relevant information not being retrieved even though it exists in the source document.

---

## Tech Stack

- Python
- `pypdf`
- NumPy
- Custom chunking and similarity search
- FastAPI (backend interface)

---

## Motivation

This project was built as part of my learning process in **LLMs and information retrieval**.  
My longer-term goal is to apply RAG concepts in practical systems, such as **ERP and business applications**, where users can ask natural language questions over structured and unstructured company data.

Building this project from scratch helped me better understand the strengths and weaknesses of RAG systems before using them in more complex environments.

---

## Status

This is a learning project and will continue to evolve as my understanding improves.

