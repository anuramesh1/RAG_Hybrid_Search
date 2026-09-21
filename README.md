# RAG Documentation Assistant

A production-quality Retrieval-Augmented Generation pipeline built with Python, LangGraph, and ChromaDB. Every concept is modularised into its own file so you can read and understand each piece in isolation.

You can use it two ways:

- **Streamlit web UI** — ask questions in the browser, rebuild the index with one click, and inspect the retrieved chunks and hybrid-search trace. See [Streamlit UI](#streamlit-ui).
- **Command-line demo scripts** — step through the chunking comparison manually. See [Demo — Character vs Semantic Chunking](#demo--character-vs-semantic-chunking).

---

## What This Project Teaches

1. **Character vs Semantic Chunking** — Why splitting at fixed byte counts destroys code blocks and mixes topics, and how splitting at Markdown `##` headers preserves meaning boundaries.
2. **Hybrid Chunking** — How a word-count gate triggers `###` sub-splitting for oversized sections, combining the best of structural and size-aware chunking.
3. **Embedding & Vector Storage** — How `sentence-transformers` converts text into fixed-size vectors, and how ChromaDB's HNSW index stores and retrieves them from disk.
4. **Hybrid Search (BM25 + Vector + RRF)** — Why keyword search (BM25) finds exact function names that semantic search misses, and how Reciprocal Rank Fusion merges both ranked lists.
5. **LangGraph RAG Pipeline** — How a `StateGraph` wires three pure-function nodes (retrieve → build_prompt → generate) into a traceable, error-handled pipeline.

---

## Setup

```bash
# 1. (Recommended) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies (use pip or pip3 depending on your Python install)
pip install -r requirements.txt

# 3. Copy the example env file and add your OpenAI key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (~90 MB) from Hugging Face. Subsequent runs use the cached copy.

---

## Add Your Documents

Place `.md` files in `docs/`. Two documents are already included:

| File | Content |
|------|---------|
| `docs/MCP_Documentation.md` | Healthcare Patient Records MCP Server guide |
| `docs/DOCUMENTATION.md` | AI Agents — Traditional Function Tools vs MCP |

After adding or changing documents, rebuild the index (sidebar button in the UI, or `python demo/ingest_structure_based.py` on the command line).

---

## Streamlit UI

The web UI wraps the same LangGraph pipeline the CLI scripts use. It is the quickest way to try the assistant.

### Run locally

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) and opens it in your browser.

### First-time use

1. **Check the API key.** The sidebar shows `API key loaded from .env` when it finds `OPENAI_API_KEY`. If it shows `No OPENAI_API_KEY found`, paste your key into the sidebar password field. That key is held in memory for the session only and is never written to disk.
2. **Build the index.** On a fresh checkout the sidebar shows `No index found`. Click **Rebuild index from docs/**. This chunks every `.md` file in `docs/` with structure-based (semantic) chunking, embeds the chunks, and stores them in `chroma_data/`. The sidebar then shows the indexed chunk count.
3. **Ask a question.** Type into **Your question** and press **Ask**, or click one of the example questions under **Try asking** in the sidebar.

### What you get back

For every question the page shows:

- **Answer** — the grounded response from `gpt-4o-mini`.
- **Citations** — the `[Source: file, Section: name]` references extracted from the answer.
- **Retrieved chunks** (expander) — the top-K chunks after Reciprocal Rank Fusion, with rank, document, section title, RRF score, and full text.
- **Hybrid search trace** (expander) — the BM25, vector, and RRF ranking output the retriever prints, so you can see exactly why each chunk was chosen.

Ask an out-of-scope question such as *"What is the rate limit on the Stripe API?"* and the assistant should reply that it could not find an answer in the documentation rather than inventing one.

### Rebuilding the index

Click **Rebuild index from docs/** whenever you add, remove, or edit files in `docs/`. The button upserts by chunk ID, so re-running on unchanged docs is safe. If you previously ingested with character splitting from the CLI, delete `chroma_data/*` first so the old chunks are not mixed with the new ones.

### Where the API key comes from

The app looks for `OPENAI_API_KEY` in this order. The first one found wins.

| Priority | Source | When to use |
|----------|--------|-------------|
| 1 | `st.secrets` (`.streamlit/secrets.toml` locally, or **Settings → Secrets** on Streamlit Community Cloud) | Deployed apps |
| 2 | `.env` in the project root | Local development |
| 3 | Sidebar password field | Quick one-off sessions with no config files |

To use a local secrets file instead of `.env`:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and set OPENAI_API_KEY = "sk-..."
```

Both `.env` and `.streamlit/secrets.toml` are gitignored.

### Deploy to Streamlit Community Cloud

1. Push the repository to GitHub. `chroma_data/`, `.env`, and `.streamlit/secrets.toml` are gitignored and will not be uploaded.
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**, and select the repo and branch.
3. Set **Main file path** to `streamlit_app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
5. Deploy. Once the app loads, click **Rebuild index from docs/** in the sidebar to build the vector store on the server.

The index lives on the app's ephemeral disk, so you will need to rebuild it after the app restarts or redeploys.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `No index found` in the sidebar | Click **Rebuild index from docs/**. |
| `Set an OpenAI API key in the sidebar before asking.` | Add `OPENAI_API_KEY` to `.env`, to `.streamlit/secrets.toml`, or paste it in the sidebar. |
| Error mentioning `invalid_api_key` or `401` | The key being sent is wrong or expired. Check the `OPENAI_API_KEY` line in `.env`, or the Secrets panel on Streamlit Cloud. |
| `Ingestion failed: No .md files found in ./docs` | Add at least one `.md` file to `docs/`. |
| First question is slow | The embedding model and ChromaDB collection load once and are cached with `st.cache_resource`. Later questions are fast. |

---

## Demo — Character vs Semantic Chunking

This is the core demo. Run the steps below in order to see exactly how chunking strategy affects RAG answer quality. Use `python` or `python3` depending on your install.

---

### Step 1 — Ingest with Character Splitting

Character splitting cuts at fixed 500-character intervals, ignoring section boundaries, code blocks, and tables.

```bash
python demo/ingest_char.py
```

Expected output: ~60–70 chunks created from the two documents.

---

### Step 2 — Query with Character-Split Chunks

```bash
python demo/user_query.py "Does this project use a remote Google Maps MCP server?"
```

**What to observe:**
- The BM25 and vector results printed before the answer will show fragmented chunks — content mid-sentence, broken code blocks, sections mixed together.
- The answer may be vague, miss key details, or reference the wrong context because the retrieved chunks don't align with meaningful document sections.

---

### Step 3 — Delete the Vector Store

```bash
rm -rf chroma_data/*
```

This clears all stored chunks and embeddings. The folder itself is preserved (it's in `.gitignore`).

---

### Step 4 — Ingest with Semantic (Structure-Based) Chunking

Semantic splitting cuts at `##` Markdown headers. Sections over 400 words are further split at `###` headers. Every chunk is a complete, self-contained section. This is the same strategy the Streamlit **Rebuild index from docs/** button uses.

```bash
python demo/ingest_structure_based.py
```

Expected output: ~18–22 chunks — far fewer but each chunk is a whole section.

---

### Step 5 — Query with Semantic Chunks

```bash
python demo/user_query.py "What Python code do I need to connect an agent to the Maps MCP server?"
```

**What to observe:**
- The retrieved chunks will be complete sections — the MCPToolset setup code, the StreamableHTTPConnectionParams block — intact and in context.
- The answer will include the actual Python code snippet and a precise citation like `[Source: DOCUMENTATION.md, Section: Modern Approach: MCP_Maps_Agent]`.

---

### Optional — Chunking Stats Comparison

To see a side-by-side statistical comparison of both strategies (broken code blocks, broken tables, average words per chunk) without touching the database:

```bash
python demo/chunking_comparison.py
```

---

### Optional — Inspect a Stored Vector

To print one stored chunk's ID, metadata, text, and raw embedding straight from ChromaDB:

```bash
python test_vectors.py
```

---

### Non-Hallucination Test

Ask a question that is not in any document. The system should refuse rather than invent an answer:

```bash
python demo/user_query.py "What is the rate limit on the Stripe API?"
```

Expected: `I could not find an answer to this question in the provided documentation.`

---

## Project Structure

```
RAG_Hybrid_Search/
├── docs/
│   ├── MCP_Documentation.md     # Source document 1
│   └── DOCUMENTATION.md         # Source document 2
├── config/
│   └── settings.py              # All constants — chunk size, model name, paths
├── chunking/
│   ├── character_splitter.py    # Fixed-interval splitting (the "bad" baseline)
│   ├── semantic_splitter.py     # Header-boundary splitting with hybrid H3 gate
│   └── comparison.py            # Runs both strategies and computes stats
├── vectorstore/
│   ├── embedder.py              # Loads SentenceTransformer, encodes chunks
│   └── store.py                 # ChromaDB PersistentClient — upsert and load
├── retrieval/
│   ├── semantic_search.py       # Pure vector search via ChromaDB
│   └── hybrid_search.py         # BM25 + vector + Reciprocal Rank Fusion
├── generation/
│   ├── prompt_builder.py        # Assembles system prompt + grounded user message
│   └── answer_generator.py      # Calls OpenAI API, extracts citations
├── pipeline/
│   ├── state.py                 # RAGState TypedDict shared across all nodes
│   ├── nodes.py                 # Three pure-function LangGraph nodes
│   └── graph.py                 # StateGraph wiring + run_query() entry point
├── demo/
│   ├── chunking_comparison.py     # stats comparison, no DB required
│   ├── ingest_char.py             # ingest with character splitting (step 1 of demo)
│   ├── ingest_structure_based.py  # ingest with semantic splitting  (step 4 of demo)
│   └── user_query.py              # query the pipeline with any question
├── .streamlit/
│   └── secrets.toml.example     # Template for a local/Cloud secrets file
├── streamlit_app.py             # Streamlit web UI over the same pipeline
├── test_vectors.py              # Prints one stored chunk + embedding from ChromaDB
├── chroma_data/                 # Auto-created by ChromaDB — in .gitignore
├── main.py                      # Docstring listing the demo steps in order
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Upgrading to Production

ChromaDB's HNSW index parameters (`m`, `ef_construct`) are chosen automatically here — good for getting started, but not tunable. To configure them explicitly, swap `vectorstore/store.py` for a Qdrant client:

```python
# vectorstore/store.py (Qdrant version)
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
)
```

No other file needs to change — `nodes.py`, `graph.py`, the demo scripts, and `streamlit_app.py` are all unaffected.
