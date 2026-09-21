from dataclasses import dataclass
from typing import List

from retrieval.semantic_search import RetrievedChunk


@dataclass
class PromptMessages:
    system: str
    user: str


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> PromptMessages:
    """
    Build a system prompt and user message from the query and retrieved chunks.

    System prompt enforces strict grounding and non-hallucination.
    User message includes all retrieved chunks with their source labels,
    followed by the question.
    """
    system = (
        "You are a documentation assistant. "
        "Answer ONLY using the information in the provided document chunks.\n\n"
        "If the answer is not present in the chunks, respond with:\n"
        "'I could not find an answer to this question in the provided documentation.'\n\n"
        "Do not infer, guess, or use external knowledge.\n\n"
        "For every claim in your answer, cite the source using this format:\n"
        "[Source: <doc_title>, Section: <section_title>]"
    )

    user_parts = ["--- Retrieved Documentation ---\n"]
    for chunk in chunks:
        doc_title = chunk.metadata.get("doc_title", "Unknown")
        section_title = chunk.metadata.get("section_title", "Unknown")
        user_parts.append(f"[Chunk {chunk.rank} - {doc_title} | {section_title}]")
        user_parts.append(chunk.content)
        user_parts.append("")

    user_parts.append("---")
    user_parts.append(f"Question: {query}")

    return PromptMessages(
        system=system,
        user="\n".join(user_parts),
    )
