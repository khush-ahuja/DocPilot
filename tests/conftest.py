import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from docpilot.embeddings import HashEmbedder  # noqa: E402
from docpilot.ingest import chunk_file, iter_files  # noqa: E402
from docpilot.store import VectorStore  # noqa: E402

STOP = {"the", "a", "an", "is", "of", "to", "do", "i", "how", "what", "can", "my", "for", "and", "in", "many", "much"}


def keywords(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


class ScriptedLLM:
    """Deterministic stand-in for a chat model. Reacts to each prompt type in docpilot.prompts."""

    def __init__(self, ungrounded_first=False, rewrite_to=None, reject_first_grade=False):
        self.calls = []
        self.ungrounded_first = ungrounded_first
        self.rewrite_to = rewrite_to
        self.reject_first_grade = reject_first_grade

    def invoke(self, prompt):
        self.calls.append(prompt)
        return SimpleNamespace(content=self._reply(prompt))

    def _reply(self, p):
        if p.startswith("You route"):
            msg = p.rsplit("Message:", 1)[1].lower()
            return '{"route": "direct"}' if re.search(r"\b(hi|hello|thanks)\b", msg) else '{"route": "retrieve"}'
        if p.startswith("You are DocPilot"):
            return "Hi! Ask me anything about your documents."
        if p.startswith("You check which"):
            if self.reject_first_grade:
                self.reject_first_grade = False
                return '{"relevant": []}'
            q = keywords(p.split("Question:", 1)[1].split("Passages:", 1)[0])
            passages = re.findall(r"\[(\d+)\] (.*?)(?=\n\n\[\d+\] |\n\nReply with JSON)", p, re.S)
            rel = [int(i) for i, t in passages if len(q & keywords(t)) >= 2]
            return f'{{"relevant": {rel}}}'
        if p.startswith("The search below"):
            return f'{{"query": "{self.rewrite_to or "rewritten query"}"}}'
        if p.startswith("Answer the question"):
            first = re.search(r"\[1\] \(.*?\)\n(.*?)(?:\n|$)", p).group(1)
            return f"{first[:120]} [1]"
        if p.startswith("Check whether"):
            if self.ungrounded_first:
                self.ungrounded_first = False
                return '{"grounded": false, "unsupported": "made-up number"}'
            return '{"grounded": true}'
        raise AssertionError(f"unexpected prompt: {p[:80]}")


@pytest.fixture
def store():
    s = VectorStore(QdrantClient(":memory:"), HashEmbedder(), "test")
    for f in iter_files([ROOT / "sample_docs"]):
        s.add(chunk_file(f, chunk_size=500, overlap=80))
    return s
