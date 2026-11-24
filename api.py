"""FastAPI service.

    uvicorn api:app --reload
    curl -F "files=@handbook.pdf" localhost:8000/ingest
    curl -H "Content-Type: application/json" -d '{"question": "What is the leave policy?"}' localhost:8000/ask
"""
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

from docpilot.ingest import SUPPORTED, chunk_file

app = FastAPI(title="DocPilot", description="Agentic RAG over your documents", version="1.0.0")


@lru_cache
def get_pilot():
    from docpilot import create_pilot
    return create_pilot()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health(p=Depends(get_pilot)):
    pilot, _ = p
    return {"status": "ok", "chunks": pilot.store.count()}


@app.get("/sources")
def sources(p=Depends(get_pilot)):
    return {"sources": p[0].store.sources()}


@app.post("/ingest")
def ingest(files: list[UploadFile], p=Depends(get_pilot)):
    pilot, settings = p
    added = {}
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            name = Path(f.filename).name
            if Path(name).suffix.lower() not in SUPPORTED:
                raise HTTPException(400, f"{name}: only {', '.join(sorted(SUPPORTED))} are supported")
            dest = Path(tmp) / name
            with dest.open("wb") as out:
                shutil.copyfileobj(f.file, out)
            pilot.store.delete_source(name)
            added[name] = pilot.store.add(chunk_file(dest, settings.chunk_size, settings.chunk_overlap))
    return {"ingested": added, "total_chunks": pilot.store.count()}


@app.post("/ask")
def ask(req: AskRequest, p=Depends(get_pilot)):
    if not req.question.strip():
        raise HTTPException(400, "question is empty")
    return p[0].ask(req.question)
