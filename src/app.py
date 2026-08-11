import streamlit as st
from agent import run_query

st.set_page_config(page_title="Document Research Assistant", layout="wide")

st.title("Agentic Document Research Assistant")
st.caption("Answers your questions from the document corpus. Asks before searching the web.")

# Streamlit re-runs this whole script on every interaction (every click,
# every text input), so anything we need to remember between reruns —
# like "we're waiting on the user to approve a web search" — has to
# live in st.session_state, not in a normal variable.
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "result" not in st.session_state:
    st.session_state.result = None

query = st.text_input("Ask a question about your documents:")

if st.button("Submit") and query:
    with st.spinner("Searching documents..."):
        result = run_query(query, web_search_approved=False)

    if result["status"] == "needs_web_search_approval":
        # stash the query so we can re-run it later WITH approval,
        # without asking the user to retype it
        st.session_state.pending_query = query
        st.session_state.result = None
    else:
        st.session_state.pending_query = None
        st.session_state.result = result

# --- case 1: waiting on the user for web search consent ---
if st.session_state.pending_query:
    st.warning("I couldn't find sufficient information in the provided documents. "
               "Would you like me to search the web for this query?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, search the web"):
            with st.spinner("Searching the web..."):
                result = run_query(st.session_state.pending_query, web_search_approved=True)
            st.session_state.result = result
            st.session_state.pending_query = None
            st.rerun()
    with col2:
        if st.button("No, don't search"):
            st.info("Okay, I won't search the web. Let me know if you'd like to rephrase your question.")
            st.session_state.pending_query = None

# --- case 2: we have a final answer to show ---
if st.session_state.result:
    result = st.session_state.result

    st.subheader("Answer")
    st.write(result["answer"])

    if result["self_corrected"]:
        st.info("Self-correction was triggered — the first answer wasn't well-supported, "
                "so this was regenerated.")

    st.subheader(f"Sources ({result['source_type']})")
    if result["source_type"] == "documents":
        for i, chunk in enumerate(result["sources"], start=1):
            with st.expander(f"Chunk {i} — {chunk['source']}, page {chunk['page']}"):
                st.write(chunk["text"])
    else:
        for i, r in enumerate(result["sources"], start=1):
            with st.expander(f"Result {i} — {r['title']}"):
                st.write(r["snippet"])
                st.write(r["url"])

    with st.expander("Agent log (what the assistant did, step by step)"):
        for step in result["log"]:
            st.write(f"- {step}")