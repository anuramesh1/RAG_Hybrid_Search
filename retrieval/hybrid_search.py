from typing import List

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.semantic_search import RetrievedChunk


def search(
    query: str,
    collection: chromadb.Collection,
    model: SentenceTransformer,
    top_k: int,
) -> List[RetrievedChunk]:
    """
    Run BM25 keyword search over all chunk texts in the collection.
    Run vector similarity search via the collection.
    Merge both ranked lists using Reciprocal Rank Fusion (RRF):
        rrf_score = 1/(rank_bm25 + 60) + 1/(rank_vector + 60)
    Return top_k results sorted by RRF score descending.
    """
    # Fetch all documents to build the BM25 corpus
    all_docs = collection.get(include=["documents", "metadatas"])
    documents: List[str] = all_docs["documents"]
    metadatas: List[dict] = all_docs["metadatas"]
    ids: List[str] = all_docs["ids"]

    if not documents:
        return []

    # BM25 keyword search
    tokenized_corpus = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

    print(f"\n{'─' * 60}")
    print(f"BM25 TOP {top_k} (keyword match)")
    print(f"{'─' * 60}")
    for i, corpus_idx in enumerate(bm25_ranked[:top_k]):
        meta = metadatas[corpus_idx]
        section = meta.get("section_title", meta.get("strategy", "?"))
        doc = meta.get("doc_title", "?")
        score = round(bm25_scores[corpus_idx], 4)
        preview = documents[corpus_idx][:80].replace("\n", " ")
        print(f"  #{i+1}  score={score:<8}  [{doc} | {section}]")
        print(f"        \"{preview}...\"")

    # Vector similarity search
    query_embedding = model.encode([query])[0].tolist()
    n_results = min(top_k * 2, len(documents))
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    vector_ids: List[str] = vector_results["ids"][0]
    vector_distances: List[float] = vector_results["distances"][0]

    # Build lookup maps
    id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}

    print(f"\n{'─' * 60}")
    print(f"VECTOR TOP {top_k} (semantic similarity)")
    print(f"{'─' * 60}")
    for i, (vid, dist) in enumerate(zip(vector_ids[:top_k], vector_distances[:top_k])):
        idx = id_to_index.get(vid, -1)
        meta = metadatas[idx] if idx >= 0 else {}
        section = meta.get("section_title", meta.get("strategy", "?"))
        doc = meta.get("doc_title", "?")
        preview = documents[idx][:80].replace("\n", " ") if idx >= 0 else ""
        print(f"  #{i+1}  dist={round(dist, 4):<8}  [{doc} | {section}]")
        print(f"        \"{preview}...\"")

    vector_rank_map = {vid: rank for rank, vid in enumerate(vector_ids)}

    # Reciprocal Rank Fusion
    rrf_scores: dict[str, float] = {}
    for rank, corpus_idx in enumerate(bm25_ranked):
        doc_id = ids[corpus_idx]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rank + 60)

    for vid, _dist in zip(vector_ids, vector_distances):
        rank = vector_rank_map[vid]
        rrf_scores[vid] = rrf_scores.get(vid, 0.0) + 1.0 / (rank + 60)

    # Return top_k by RRF score
    sorted_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    print(f"\n{'─' * 60}")
    print(f"AFTER RRF — {top_k} chunks sent to LLM")
    print(f"{'─' * 60}")
    chunks: List[RetrievedChunk] = []
    for rank, doc_id in enumerate(sorted_ids):
        idx = id_to_index[doc_id]
        meta = metadatas[idx]
        section = meta.get("section_title", meta.get("strategy", "?"))
        doc = meta.get("doc_title", "?")
        rrf = round(rrf_scores[doc_id], 6)
        preview = documents[idx][:80].replace("\n", " ")
        print(f"  #{rank+1}  rrf={rrf:<10}  [{doc} | {section}]")
        print(f"        \"{preview}...\"")
        chunks.append(RetrievedChunk(
            content=documents[idx],
            metadata=meta,
            score=rrf_scores[doc_id],
            rank=rank + 1,
        ))

    print(f"{'─' * 60}\n")
    return chunks
