import os
from pypdf import PdfReader
from config import DATA_DIR


def load_documents():
    """
    Reads every PDF in the data folder and returns a list of pages.
    Each page is a dict so we keep track of which file and page
    number it came from — we need that later for citations.
    """
    documents = []

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"No PDFs found in {DATA_DIR}/. Add some and try again.")

    for filename in pdf_files:
        filepath = os.path.join(DATA_DIR, filename)
        reader = PdfReader(filepath)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()

            # skip pages that are blank or just images with no text
            if not text or not text.strip():
                continue

            documents.append({
                "source": filename,
                "page": page_number,
                "text": text
            })

    print(f"Loaded {len(documents)} pages from {len(pdf_files)} PDFs.")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print("\nSample page:")
    print(docs[0])