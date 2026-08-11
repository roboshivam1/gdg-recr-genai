import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from ingest import load_documents


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Splits text into overlapping chunks.
    Overlap matters here — without it, a sentence that starts at the
    end of one chunk and finishes in the next would get cut in half,
    and neither chunk alone would make sense on its own.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # step back a bit so chunks overlap

    return chunks


def build_index():
    print("Loading PDFs...")
    pages = load_documents()

    print("Chunking text...")
    all_chunks = []
    for page in pages:
        for chunk in chunk_text(page["text"]):
            all_chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })
    print(f"Created {len(all_chunks)} chunks.")

    print("Loading embedding model (first run downloads it, be patient)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")
    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # wipe any old collection so re-running this script doesn't duplicate data
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks]
    )

    print(f"Done. Indexed {len(all_chunks)} chunks into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    build_index()