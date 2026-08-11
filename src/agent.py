from anthropic import Anthropic

from config import ANTHROPIC_API_KEY, CLAUDE_MODEL
from rag import answer_from_documents, retrieve, generate_answer, format_context
from web_search import answer_from_web

claude = Anthropic(api_key=ANTHROPIC_API_KEY)


def verify_answer(query, answer, context):
    """
    Asks Claude to check its own earlier answer against the context
    it was given. This is a separate call on purpose — asking the
    same call to both answer AND grade itself tends to just agree
    with whatever it already said. A fresh call with only the
    question "is this actually supported?" is a more honest check.
    """
    prompt = f"""You are verifying an AI-generated answer against its source context.

Context:
{context}

Question: {query}

Answer to verify: {answer}

Is this answer fully supported by the context above? Reply with exactly
one word: YES or NO.
"""

    response = claude.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )

    verdict = response.content[0].text.strip().upper()
    return "YES" in verdict


def run_query(query, web_search_approved=False):
    """
    The main orchestration function — this is what the Streamlit app will call. It's basically a flowchart of the whole
    assignment:

      1. Try the documents first, always.
      2. If that's not enough and the user hasn't approved web search,
         stop and ask.
      3. If web search is approved, use it instead.
      4. Verify whichever answer we ended up with.
      5. If verification fails, retry retrieval once and regenerate.
    """
    log = []  # we'll show this in the UI so self-correction is visible

    # Step 1: try the documents
    doc_result = answer_from_documents(query)
    log.append("Searched document corpus.")

    if doc_result["sufficient"]:
        answer = doc_result["answer"]
        context = format_context(doc_result["chunks"])
        source_type = "documents"
        chunks_or_results = doc_result["chunks"]
    else:
        log.append("Documents did not contain enough information.")

        if not web_search_approved:
            # stop here — the app will ask the user for permission
            return {
                "status": "needs_web_search_approval",
                "log": log
            }

        # Step 3: web search approved, use it
        log.append("User approved web search. Searching the web.")
        web_result = answer_from_web(query)
        answer = web_result["answer"]
        context = "\n".join(r["snippet"] for r in web_result["results"])
        source_type = "web"
        chunks_or_results = web_result["results"]

    # Step 4: verify
    log.append("Running self-verification check.")
    is_supported = verify_answer(query, answer, context)

    if not is_supported:
        log.append("Self-correction triggered: answer was not well-supported. Retrying.")

        if source_type == "documents":
            # try retrieval again, asking for more chunks this time
            chunks = retrieve(query, top_k=8)
            answer = generate_answer(query, chunks)
            context = format_context(chunks)
            chunks_or_results = chunks
        else:
            # try the web again with a slightly reworded query
            web_result = answer_from_web(query + " explained")
            answer = web_result["answer"]
            chunks_or_results = web_result["results"]

        log.append("Regenerated answer after self-correction.")
    else:
        log.append("Self-verification passed on the first attempt.")

    return {
        "status": "answered",
        "answer": answer,
        "source_type": source_type,
        "sources": chunks_or_results,
        "self_corrected": not is_supported,
        "log": log
    }


# quick manual test
if __name__ == "__main__":
    result = run_query("What is this document about?")
    print(result)