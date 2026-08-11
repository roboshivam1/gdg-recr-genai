import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

from config import (
    CHROMA_DIR, COLLECTION_NAME, ANTHROPIC_API_KEY, CLAUDE_MODEL
)

# load these once, not on every call — loading the model or reconnecting
# to chroma on every single query would be slow
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
claude = Anthropic(api_key=ANTHROPIC_API_KEY)


def retrieve(query, top_k=5):
    """
    Turns the query into an embedding and finds the most similar
    chunks stored in ChromaDB. Returns them with their source/page info.
    """
    query_embedding = embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    chunks = []
    for text, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text": text,
            "source": meta["source"],
            "page": meta["page"]
        })

    return chunks


def format_context(chunks):
    """
    Turns retrieved chunks into a labeled block of text we can drop
    into the prompt, so Claude can point back to exactly which chunk
    it used.
    """
    formatted = ""
    for i, chunk in enumerate(chunks, start=1):
        formatted += f"[Chunk {i} - {chunk['source']}, page {chunk['page']}]\n{chunk['text']}\n\n"
    return formatted


def generate_answer(query, chunks):
    """
    Asks Claude to answer the query using ONLY the retrieved chunks.
    We're explicit that it should say so if the context isn't enough —
    that's what lets the rest of the app decide whether to offer a
    web search.
    """
    context = format_context(chunks)

    prompt = f"""You are a research assistant. Answer the question using ONLY the context below.

Context:
{context}

Question: {query}

Rules:
- Only use information from the context above, nothing from your own knowledge.
- If you use a chunk, cite it like this: (Source: filename, page X)
- If the context does not contain enough information to answer the question,
  respond with exactly: NOT_ENOUGH_CONTEXT
"""

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def answer_from_documents(query):
    """
    Full pipeline: retrieve chunks, ask Claude, return both the
    answer and the chunks used (the UI will want to show these).
    """
    chunks = retrieve(query)
    answer = generate_answer(query, chunks)
    is_sufficient = "NOT_ENOUGH_CONTEXT" not in answer

    return {
        "answer": answer,
        "chunks": chunks,
        "sufficient": is_sufficient
    }


# quick manual test
if __name__ == "__main__":
    result = answer_from_documents("What is this document about?")
    print("Sufficient:", result["sufficient"])
    print("Answer:", result["answer"])