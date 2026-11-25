"""Benchmark on a public dataset: plain RAG vs the DocPilot agent on SQuAD 2.0.

SQuAD 2.0 mixes answerable questions with *unanswerable* ones written to look answerable from the
same paragraphs. That makes it a good test of the thing DocPilot is built for: answering when the
documents support it, and saying "not found" instead of making something up when they don't.

    python eval/squad_bench.py --articles 8 --answerable 36 --unanswerable 24

Setup
- Corpus: every paragraph of N Wikipedia articles from the SQuAD 2.0 dev set (one chunk per paragraph),
  indexed in an in-memory Qdrant collection with FastEmbed (bge-small).
- Both systems use the same retriever, top-k, answer model and answer prompt. The only difference is
  DocPilot's routing, chunk grading, query rewriting and grounding check.
- Judge: a larger model (default llama-3.3-70b-versatile) labels each answer correct / incorrect /
  abstained against the official SQuAD answers.

Writes eval/squad_results.json and eval/squad_samples.jsonl (every answer, for error analysis).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docpilot import prompts  # noqa: E402
from docpilot.config import Settings  # noqa: E402
from docpilot.graph import DocPilot, _context  # noqa: E402
from docpilot.ingest import Chunk  # noqa: E402
from docpilot.llm import parse_json  # noqa: E402

JUDGE = """You grade a question-answering system that must answer only from given documents.

Question: {question}
Accepted answers: {gold}
System answer: {answer}

Label the system answer:
- "correct": it gives an answer consistent with one of the accepted answers
- "abstained": it says the information is not in the documents / it could not find it
- "incorrect": it gives an answer that does not match any accepted answer
If accepted answers is "NONE", the question has no answer in the documents: label "abstained" if the system
declined, otherwise "incorrect".
Reply with JSON only: {{"label": "correct" | "incorrect" | "abstained"}}"""


class CountingLLM:
    """Wraps a chat model and counts calls and tokens."""

    def __init__(self, llm):
        self.llm, self.calls, self.tokens = llm, 0, 0

    def invoke(self, prompt):
        out = self.llm.invoke(prompt)
        self.calls += 1
        usage = (getattr(out, "response_metadata", {}) or {}).get("token_usage", {}) or {}
        self.tokens += usage.get("total_tokens", 0)
        return out

    def snapshot(self):
        return self.calls, self.tokens


def make_llm(model: str, temperature: float = 0):
    from langchain_groq import ChatGroq

    return ChatGroq(model=model, temperature=temperature, max_retries=10)


def load_squad(n_articles: int, n_ans: int, n_unans: int, seed: int):
    from datasets import load_dataset

    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    rng = random.Random(seed)
    titles = sorted(set(ds["title"]))
    picked = set(rng.sample(titles, n_articles))
    rows = [r for r in ds if r["title"] in picked]
    paragraphs = sorted({(r["title"], r["context"]) for r in rows})
    chunks = [Chunk(text=ctx, source=title, chunk_index=i) for i, (title, ctx) in enumerate(paragraphs)]
    ans = [r for r in rows if r["answers"]["text"]]
    unans = [r for r in rows if not r["answers"]["text"]]
    qs = rng.sample(ans, min(n_ans, len(ans))) + rng.sample(unans, min(n_unans, len(unans)))
    rng.shuffle(qs)
    return chunks, qs, sorted(picked)


def build_store(chunks, embedding_model: str):
    from qdrant_client import QdrantClient

    from docpilot.embeddings import FastEmbedEmbedder
    from docpilot.store import VectorStore

    store = VectorStore(QdrantClient(":memory:"), FastEmbedEmbedder(embedding_model), "squad")
    store.add(chunks)
    return store


def summarize(samples):
    ans = [s for s in samples if s["answerable"]]
    un = [s for s in samples if not s["answerable"]]

    def rate(xs, label):
        return round(sum(x["label"] == label for x in xs) / max(len(xs), 1), 3)

    return {
        "answerable_correct": rate(ans, "correct"),
        "answerable_wrong": rate(ans, "incorrect"),
        "answerable_abstained": rate(ans, "abstained"),
        "unanswerable_abstained": rate(un, "abstained"),
        "unanswerable_hallucinated": rate(un, "incorrect"),
        "overall_accuracy": round((sum(x["label"] == "correct" for x in ans)
                                   + sum(x["label"] == "abstained" for x in un)) / max(len(samples), 1), 3),
        "avg_latency_s": round(sum(s["latency_s"] for s in samples) / len(samples), 2),
        "llm_calls_per_q": round(sum(s["calls"] for s in samples) / len(samples), 2),
        "tokens_per_q": round(sum(s["tokens"] for s in samples) / len(samples)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--articles", type=int, default=8)
    p.add_argument("--answerable", type=int, default=36)
    p.add_argument("--unanswerable", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="llama-3.1-8b-instant")
    p.add_argument("--judge", default="llama-3.3-70b-versatile")
    p.add_argument("--out", default="eval/squad_results.json")
    a = p.parse_args()

    settings = Settings()
    chunks, questions, titles = load_squad(a.articles, a.answerable, a.unanswerable, a.seed)
    print(f"{len(chunks)} paragraphs from {len(titles)} articles; {len(questions)} questions")
    store = build_store(chunks, settings.embedding_model)

    llm = CountingLLM(make_llm(a.model))
    judge = make_llm(a.judge)
    pilot = DocPilot(llm, store, settings.top_k, settings.max_rewrites)

    def run_naive(q):
        hits = store.search(q, k=settings.top_k)
        return llm.invoke(prompts.GENERATE.format(context=_context(hits), question=q)).content.strip()

    def run_agent(q):
        return pilot.ask(q)["answer"]

    samples = {"plain_rag": [], "docpilot": []}
    for i, r in enumerate(questions, 1):
        gold = r["answers"]["text"]
        for name, fn in (("plain_rag", run_naive), ("docpilot", run_agent)):
            c0, t0_tok = llm.snapshot()
            t0 = time.perf_counter()
            answer = fn(r["question"])
            latency = time.perf_counter() - t0
            c1, t1_tok = llm.snapshot()
            verdict = parse_json(judge.invoke(JUDGE.format(
                question=r["question"], gold=" | ".join(sorted(set(gold))) if gold else "NONE",
                answer=answer)).content, {})
            label = verdict.get("label", "incorrect")
            label = label if label in ("correct", "incorrect", "abstained") else "incorrect"
            samples[name].append({"id": r["id"], "title": r["title"], "question": r["question"], "gold": gold,
                                  "answerable": bool(gold), "answer": answer, "label": label,
                                  "latency_s": round(latency, 2), "calls": c1 - c0, "tokens": t1_tok - t0_tok})
        print(f"[{i}/{len(questions)}] plain={samples['plain_rag'][-1]['label']:<10} "
              f"docpilot={samples['docpilot'][-1]['label']:<10} {r['question'][:60]}")

    report = {name: summarize(s) for name, s in samples.items()}
    meta = {"dataset": "SQuAD 2.0 dev (rajpurkar/squad_v2)", "articles": titles, "paragraphs": len(chunks),
            "questions": len(questions), "answerable": sum(bool(q["answers"]["text"]) for q in questions),
            "answer_model": f"groq/{a.model}", "judge_model": f"groq/{a.judge}",
            "embeddings": settings.embedding_model, "top_k": settings.top_k, "seed": a.seed}
    Path(a.out).write_text(json.dumps({"config": meta, "results": report}, indent=2))
    with open(Path(a.out).with_name("squad_samples.jsonl"), "w") as f:
        for name, s in samples.items():
            for x in s:
                f.write(json.dumps({"system": name, **x}) + "\n")

    cols = list(next(iter(report.values())).keys())
    print("\n" + json.dumps(meta, indent=2))
    print(f"\n{'metric':<28}{'plain_rag':>12}{'docpilot':>12}")
    for c in cols:
        print(f"{c:<28}{report['plain_rag'][c]:>12}{report['docpilot'][c]:>12}")


if __name__ == "__main__":
    main()
