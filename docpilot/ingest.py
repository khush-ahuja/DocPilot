"""Load PDF / Markdown / text files and split them into overlapping chunks."""
import re
from dataclasses import dataclass
from pathlib import Path

SUPPORTED = {".pdf", ".md", ".txt"}


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    chunk_index: int = 0


def load_file(path: str | Path) -> list[tuple[str, int | None]]:
    """Returns (text, page_number) pairs; page is None for non-PDF files."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return [(p.extract_text() or "", i + 1) for i, p in enumerate(PdfReader(str(path)).pages)]
    if path.suffix.lower() in {".md", ".txt"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    raise ValueError(f"unsupported file type: {path.suffix}")


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Paragraph-aware splitter: packs paragraphs up to chunk_size characters,
    hard-splits oversized paragraphs, and carries `overlap` characters forward."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces = []
    for p in paras:
        while len(p) > chunk_size:
            cut = p.rfind(" ", 0, chunk_size)
            cut = cut if cut > chunk_size // 2 else chunk_size
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)

    chunks, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 2 > chunk_size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cut = tail.find(" ")
            cur = (tail[cut + 1:] if cut >= 0 else tail) + "\n\n" + piece if overlap else piece
        else:
            cur = f"{cur}\n\n{piece}" if cur else piece
    if cur:
        chunks.append(cur)
    return chunks


def chunk_file(path: str | Path, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    name = Path(path).name
    out = []
    for text, page in load_file(path):
        for t in split_text(text, chunk_size, overlap):
            out.append(Chunk(text=t, source=name, page=page, chunk_index=len(out)))
    return out


def iter_files(paths) -> list[Path]:
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED)
        elif p.suffix.lower() in SUPPORTED:
            files.append(p)
    return files
