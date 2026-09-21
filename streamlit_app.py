"""
Streamlit UI for the hybrid-search RAG documentation assistant.

Local:
    streamlit run streamlit_app.py

Deploy (Streamlit Community Cloud):
    push to GitHub -> share.streamlit.io -> main file: streamlit_app.py
    -> Settings > Secrets -> OPENAI_API_KEY = "sk-..."
"""
import io
import os
import sys
from contextlib import redirect_stdout

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Docs RAG Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# API key resolution, before the pipeline is imported.
# answer_generator reads OPENAI_API_KEY at call time, and its load_dotenv() will
# not overwrite a value already in the environment, so secrets win over .env.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv


def _key_from_secrets() -> str:
    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        return ""   # no secrets.toml — normal when running locally


_secret_key = _key_from_secrets()
if _secret_key:
    os.environ["OPENAI_API_KEY"] = _secret_key
load_dotenv()                            # local .env; will not override the secret
_key = os.getenv("OPENAI_API_KEY", "")

from chunking.semantic_splitter import split_by_headers          # noqa: E402
from config.settings import (                                     # noqa: E402
    CHROMA_PATH,
    COLLECTION_NAME,
    DOCS_PATH,
    EMBED_MODEL,
    SEMANTIC_MAX_WORDS,
    TOP_K,
)
from vectorstore.embedder import embed_chunks, load_embedding_model   # noqa: E402
from vectorstore.store import (                                       # noqa: E402
    get_or_create_collection,
    load_collection,
    upsert_chunks,
)
import pipeline.nodes as nodes                                    # noqa: E402
from pipeline.graph import build_rag_graph                        # noqa: E402


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str):
    return load_embedding_model(model_name)


@st.cache_resource(show_spinner=False)
def get_collection(chroma_path: str, collection_name: str):
    return load_collection(chroma_path, collection_name)


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_rag_graph()


# retrieve_node reloads the embedding model and reopens Chroma on every call.
# Point it at the cached versions so a 90 MB model isn't re-read per question.
nodes.load_embedding_model = get_embedding_model
nodes.load_collection = get_collection


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def index_status():
    """Return (ready, chunk_count). Never raises."""
    try:
        return True, get_collection(CHROMA_PATH, COLLECTION_NAME).count()
    except Exception:
        return False, 0


def build_index() -> int:
    """Ingest docs/*.md with structure-based chunking. Returns chunks stored."""
    md_files = sorted(
        os.path.join(DOCS_PATH, f)
        for f in os.listdir(DOCS_PATH)
        if f.endswith(".md")
    )
    if not md_files:
        raise RuntimeError(f"No .md files found in ./{DOCS_PATH}")

    all_chunks = []
    for path in md_files:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        all_chunks.extend(
            split_by_headers(text, os.path.basename(path), SEMANTIC_MAX_WORDS)
        )

    model = get_embedding_model(EMBED_MODEL)
    embeddings = embed_chunks(all_chunks, model)
    collection = get_or_create_collection(CHROMA_PATH, COLLECTION_NAME)
    upsert_chunks(collection, all_chunks, embeddings)

    get_collection.clear()   # reopen the collection on next access
    return len(all_chunks)


def run_pipeline(query: str):
    """Invoke the graph, capturing the hybrid-search trace it prints to stdout."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        state = get_graph().invoke({
            "query": query,
            "retrieved_chunks": [],
            "prompt": None,
            "answer": None,
            "error": "",
        })
    return state, buffer.getvalue()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
ready, chunk_count = index_status()

with st.sidebar:
    st.header("Status")

    if _key:
        source = "Secrets" if _secret_key else ".env"
        st.success(f"API key loaded from {source}")
    else:
        st.warning("No OPENAI_API_KEY found")
        entered = st.text_input(
            "OpenAI API key",
            type="password",
            help="Used for this session only; nothing is written to disk.",
        )
        if entered:
            os.environ["OPENAI_API_KEY"] = entered
            _key = entered

    if ready:
        st.metric("Indexed chunks", chunk_count)
    else:
        st.error("No index found")

    st.caption(f"Embeddings: `{EMBED_MODEL}`")
    st.caption(f"Generator: `gpt-4o-mini`")
    st.caption(f"Retrieval: BM25 + vector, RRF fused, top {TOP_K}")

    st.divider()
    if st.button("Rebuild index from docs/", use_container_width=True):
        with st.spinner("Chunking, embedding, upserting…"):
            try:
                stored = build_index()
                st.success(f"Indexed {stored} chunks")
                st.rerun()
            except Exception as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    st.caption("Try asking")
    for example in (
        "How does FastMCP register a tool with the MCP server?",
        "What is the difference between Stdio and Streamable HTTP?",
        "What is the rate limit on the Stripe API?",
    ):
        if st.button(example, key=f"ex_{example}", use_container_width=True):
            st.session_state.query = example


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.title("📚 Documentation Assistant")
st.caption(
    "Hybrid retrieval (BM25 + embeddings, fused with Reciprocal Rank Fusion) "
    "over your Markdown docs. Answers are grounded in retrieved chunks and cited."
)

if not ready:
    st.info(
        "The vector store is empty. Use **Rebuild index from docs/** in the sidebar "
        "to ingest the Markdown files, then ask a question."
    )

with st.form("ask"):
    query = st.text_input(
        "Your question",
        value=st.session_state.get("query", ""),
        placeholder="How does FastMCP register a tool with the MCP server?",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and query.strip():
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Set an OpenAI API key in the sidebar before asking.")
    elif not ready:
        st.error("Build the index first.")
    else:
        st.session_state.query = query
        with st.spinner("Retrieving and generating…"):
            state, trace = run_pipeline(query)

        if state.get("error"):
            message = state["error"]
            st.error(message)
            if "invalid_api_key" in message or "401" in message:
                st.caption(
                    "The key being sent is invalid. Check Secrets on Streamlit Cloud, "
                    "or the OPENAI_API_KEY line in your local .env."
                )
        else:
            answer = state["answer"]
            chunks = state["retrieved_chunks"]

            st.subheader("Answer")
            st.markdown(answer.text)

            st.subheader("Citations")
            if answer.citations:
                for citation in answer.citations:
                    st.markdown(f"- `{citation}`")
            else:
                st.caption("No citations found in the answer.")

            with st.expander(f"Retrieved chunks ({answer.chunks_used})"):
                for chunk in chunks:
                    meta = chunk.metadata
                    doc = meta.get("doc_title", "?")
                    section = meta.get("section_title", meta.get("strategy", "?"))
                    st.markdown(f"**#{chunk.rank} · {doc} — {section}**")
                    st.caption(f"RRF score {chunk.score:.6f}")
                    st.text(chunk.content)
                    st.divider()

            with st.expander("Hybrid search trace (BM25 / vector / RRF)"):
                st.code(trace or "(no trace captured)", language="text")
