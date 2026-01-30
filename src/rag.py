import numpy as np
from groq import Groq
import json
from src.retriever import retrieve
from src.config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def format_prompt(context: str, question: str) -> str:
    return f"""
You are an information extraction system.

RULES:
- Use ONLY the sentences in CONTEXT.
- Copy sentences verbatim.
- Do NOT add explanations.
- Do NOT paraphrase.
- If nothing answers the question, return:
  {{ "answer": [], "found": false }}

OUTPUT FORMAT (JSON ONLY):
{{
  "answer": [string, string, ...],
  "found": true | false
}}

CONTEXT:
{context}

QUESTION:
{question}

JSON ANSWER:
"""

def rag_answer(query: str, k:int= 1, vectorstore_path="data/vectorstore.json") -> str:

    """
    Full RAG pipeline:
    1. Retrieve chunks
    2. Build prompt with context
    3. Generate answer with LLaMA 3.1 (70B)
    """

    results = retrieve(query, vectorstore_path=vectorstore_path, k=k)

    context = "\n\n".join(
        f'(Source: {r["doc_name"]} — Chunk {r["chunk_id"]})\n{r["text"]}'
        for r in results
)

    prompt = format_prompt(context, query)

    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You answer ONLY from the given context."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.1, top_p=0.3, max_tokens=200
)

    raw_answer = completion.choices[0].message.content

    try:
        parsed = json.loads(raw_answer)
    except json.JSONDecodeError:
        parsed = {
            "answer": [],
            "found": False
        }
    if "Not in context" in raw_answer:
        return {
            "query": query,
            "answer": "Not in context."
        }

    else:
        return {
            "query": query,
            "answer": parsed["answer"],
            "found": parsed["found"],
            "sources": [
                f'{r["doc_name"]} — Chunk {r["chunk_id"]}'
                for r in results
    ]
}


