from duckduckgo_search import DDGS
from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

claude = Anthropic(api_key=ANTHROPIC_API_KEY)


def search_web(query, max_results=5):
    """
    Runs a web search and returns results with title, url, and snippet.
    This function only ever runs when something upstream has already
    gotten the user's explicit yes — it has no consent logic of its
    own, that lives in the app layer (Step 7).
    """
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", "")
            })
    return results


def format_web_context(results):
    """
    Same idea as format_context() in rag.py — label each result so
    Claude can point back to exactly which one it used.
    """
    formatted = ""
    for i, r in enumerate(results, start=1):
        formatted += f"[Web Result {i} - {r['title']}]\nURL: {r['url']}\n{r['snippet']}\n\n"
    return formatted


def generate_web_answer(query, results):
    """
    Same pattern as generate_answer() in rag.py, but grounded in web
    results instead of document chunks, and citing URLs instead of
    filenames/pages.
    """
    context = format_web_context(results)

    prompt = f"""You are a research assistant. Answer the question using ONLY the web results below.

Web Results:
{context}

Question: {query}

Rules:
- Only use information from the web results above.
- Cite sources using their URL, like this: (Source: https://example.com)
- Be clear this answer came from the web, not from the user's documents.
"""

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text


def answer_from_web(query):
    """
    Full web-search pipeline. Only ever call this after consent
    has already been given — see app.py in Step 7.
    """
    results = search_web(query)
    answer = generate_web_answer(query, results)

    return {
        "answer": answer,
        "results": results
    }


# quick manual test — this file can be tested in isolation without
# going through the consent flow, since consent is enforced by the
# app, not by this module
if __name__ == "__main__":
    result = answer_from_web("What is Retrieval-Augmented Generation?")
    print(result["answer"])