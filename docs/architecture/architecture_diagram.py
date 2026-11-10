"""Render the RAG Hybrid Search architecture diagram (rag-hybrid-search-architecture.svg/.png).

Two stories: building the index, and answering a question.

    python docs/architecture/architecture_diagram.py    # PNG needs rsvg-convert

This folder is safe inside docs/: ingestion only reads docs/*.md, not subfolders.
"""
import base64
import shutil
import subprocess
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rag-hybrid-search-architecture"
W, H = 1600, 850

BLUE, PURPLE, GREEN, GREY, INK, TEAL = "#1C7ED6", "#7048E8", "#2B8A3E", "#495057", "#212529", "#0B7285"
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
# Simple Icons logos are single-colour SVGs; these are the brand colours.
BRAND = {"openai": "#000000", "huggingface": "#FF9D00", "streamlit": "#FF4B4B", "markdown": "#000000"}
parts: list[str] = []


def logo(name: str) -> str:
    svg = HERE / "logos" / f"{name}.svg"
    if svg.exists():
        coloured = svg.read_text().replace("<svg ", f'<svg fill="{BRAND[name]}" ', 1)
        return "data:image/svg+xml;base64," + base64.b64encode(coloured.encode()).decode()
    return "data:image/png;base64," + base64.b64encode((HERE / "logos" / f"{name}.png").read_bytes()).decode()


def text(x, y, lines, size=15, color=INK, weight="normal", anchor="middle"):
    lines = [lines] if isinstance(lines, str) else lines
    spans = "".join(f'<tspan x="{x}" dy="{0 if i == 0 else size + 4}">{escape(ln)}</tspan>'
                    for i, ln in enumerate(lines))
    parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{color}" '
                 f'font-weight="{weight}" text-anchor="{anchor}">{spans}</text>')


def panel(x, y, w, h, title, fill, stroke, size=18):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="2.5"/>')
    text(x + 20, y + 32, title, size=size, color=stroke, weight="bold", anchor="start")


def image(name, x, y, size):
    parts.append(f'<image href="{logo(name)}" x="{x}" y="{y}" width="{size}" height="{size}"/>')


def icon(name, cx, cy, label, size=56):
    image(name, cx - size / 2, cy - size / 2, size)
    text(cx, cy + size / 2 + 22, label, size=15)


def step(x, y, w, h, title, sub, color, fill):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
                 f'stroke="{color}" stroke-width="2.2"/>')
    sub = [sub] if isinstance(sub, str) else sub
    top = y + h / 2 - (len(sub) * 18) / 2 + 2
    text(x + w / 2, top, title, size=17, color=color, weight="bold")
    text(x + w / 2, top + 22, sub, size=14)


def arrow(d, color, label=None, at=None, anchor="middle"):
    parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.8" '
                 f'marker-end="url(#a-{color[1:]})"/>')
    if label:
        parts.append(f'<text x="{at[0]}" y="{at[1]}" font-family="{FONT}" font-size="15" '
                     f'font-weight="bold" text-anchor="{anchor}" stroke="white" stroke-width="6" '
                     f'stroke-linejoin="round" paint-order="stroke" fill="{color}">{escape(label)}</text>')


# ------------------------------------------------------------------ title
text(W / 2, 48, "RAG Hybrid Search  ·  a documentation assistant", size=30, weight="bold")
text(W / 2, 80, "Keyword search + vector search, merged, then answered with citations. "
     "Everything runs locally except the final LLM call.", size=17, color=GREY)

# ------------------------------------------------------------------ story 1: building the index
panel(30, 105, 1540, 230, "①  Building the index", "#F4FCF5", GREEN)

icon("markdown", 90, 225, ["docs/*.md"])
arrow("M125 225 L 150 225", GREEN)
step(150, 170, 340, 130, "Chunking",
     ["Structure: split at ##, then ### if over 400 words", "Character: every 500 chars, 50 overlap",
      "(built to compare the two)"], GREEN, "#EBFBEE")
arrow("M490 235 L 540 235", GREEN)
step(540, 170, 250, 130, "Embedding", ["all-MiniLM-L6-v2", "sentence-transformers", "runs locally · 384 dimensions"],
     GREEN, "#EBFBEE")
image("huggingface", 552, 180, 30)
arrow("M790 235 L 830 235", GREEN)

parts.append(f'<rect x="830" y="150" width="270" height="170" rx="12" fill="white" stroke="{TEAL}" stroke-width="2.2"/>')
image("chroma", 929, 155, 72)
text(965, 240, "ChromaDB", size=17, color=TEAL, weight="bold")
text(965, 264, ["collection rag_docs", "HNSW index, on disk (chroma_data/)"], size=14)

text(1140, 180, "How it is started", size=15, color=GREY, weight="bold", anchor="start")
text(1140, 206, ["Streamlit “Rebuild index” button", "demo/ingest_structure_based.py", "demo/ingest_char.py"],
     size=14, anchor="start")

# ------------------------------------------------------------------ story 2: answering a question
panel(30, 355, 1540, 480, "②  Answering a question", "#F8FBFF", BLUE)

icon("users", 75, 590, ["User"], size=64)
arrow("M110 590 L 155 590", BLUE)
icon("streamlit", 190, 590, ["Streamlit UI", "or CLI script"])

panel(295, 395, 1035, 430, "", "#FFFFFF", PURPLE)
image("langgraph", 312, 405, 34)
text(356, 428, "LangGraph StateGraph", size=17, color=PURPLE, weight="bold", anchor="start")
arrow("M222 590 L 308 590", BLUE, "question", (258, 578))

panel(310, 450, 700, 320, "1 · retrieve_node — hybrid search", "#F3F0FF", PURPLE, size=16)
step(330, 490, 320, 80, "BM25 keyword search", ["rank-bm25 · finds exact words", "like function names"], BLUE, "#E7F5FF")
step(680, 490, 310, 80, "Vector search", ["question embedded with MiniLM,", "closest chunks by meaning"], TEAL, "#E3FAFC")
arrow("M490 570 L 490 640", BLUE)
arrow("M835 570 L 835 640", TEAL)
step(400, 640, 520, 90, "Reciprocal Rank Fusion",
     ["score = 1/(60 + BM25 rank) + 1/(60 + vector rank)", "chunks found by both searches rise to the top"],
     PURPLE, "#FFFFFF")

arrow("M850 320 L 850 345 L 620 345 L 620 488", TEAL, "all chunk texts", (626, 440), anchor="start")
arrow("M950 320 L 950 488", TEAL, "10 closest chunks", (958, 440), anchor="start")

step(1040, 500, 270, 80, "2 · build_prompt_node", ["chunks + question", "+ citation rules"], PURPLE, "#F3F0FF")
step(1040, 650, 270, 80, "3 · generate_node", "calls the LLM", PURPLE, "#F3F0FF")
arrow("M920 685 L 1022 685 L 1022 540 L 1038 540", PURPLE, "top 5", (966, 675))
arrow("M1175 580 L 1175 648", PURPLE)
text(1175, 470, "a retrieval error ends the run here", size=13.5, color=GREY)

arrow("M1310 690 L 1410 690", GREY, "prompt", (1370, 680))
icon("openai", 1450, 690, ["OpenAI", "gpt-4o-mini"])

arrow("M1175 730 L 1175 800 L 190 800 L 190 672", BLUE, "answer + citations: [Source: file, Section: title]", (740, 790))

# ------------------------------------------------------------------ write
markers = "".join(
    f'<marker id="a-{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
    f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
    for c in (BLUE, PURPLE, GREEN, GREY, TEAL))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
       f'<defs>{markers}</defs><rect width="{W}" height="{H}" fill="white"/>' + "".join(parts) + "</svg>")
OUT.with_suffix(".svg").write_text(svg)
print(f"wrote {OUT.with_suffix('.svg').name}")
if shutil.which("rsvg-convert"):
    subprocess.run(["rsvg-convert", "-z", "2", "-o", str(OUT.with_suffix(".png")),
                    str(OUT.with_suffix(".svg"))], check=True)
    print(f"wrote {OUT.with_suffix('.png').name} ({W * 2}x{H * 2})")
