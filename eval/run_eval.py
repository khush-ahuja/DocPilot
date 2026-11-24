"""Compare the DocPilot agent against plain retrieve-then-answer RAG.

    python ingest.py sample_docs/
    python eval/run_eval.py --qa eval/qa.jsonl

Metrics (per system):
  retrieval_hit   share of answerable questions whose source document is in the top-k results
  correct         share of answers an LLM judge marks as matching the reference answer
  abstained_ok    share of unanswerable questions where the system said it could not find the answer
  cited           share of answerable questions whose answer contains a [n] citation
  avg_latency_s   mean seconds per question
Writes eval/results.json.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docpilot import create_pilot, prompts  # noqa: E402
from docpilot.graph import _context  # noqa: E402
from docpilot.llm import get_llm, parse_json  # noqa: E402

JUDGE = """You grade a question-answering system.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Is the system answer correct and consistent with the reference (extra correct detail is fine)?
Reply with JSON only: {{"correct": true or false}}"""

ABSTAIN = re.compile(
    r"(couldn'?t|could not|cannot|can't|unable to|no information|not (?:found|mentioned|stated|in the))", re.I)


def naive_rag(llm, store, question, k):
    hits = store.search(question, k=k)
    answer = llm.invoke(prompts.GENERATE.format(context=_context(hits), question=question)).content.strip()
    return answer, hits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qa", default="eval/qa.jsonl")
    p.add_argument("--out", default="eval/results.json")
    args = p.parse_args()

    pilot, settings = create_pilot()
    llm = get_llm(settings)
    rows = [json.loads(line) for line in Path(args.qa).read_text().splitlines() if line.strip()]

    def judge(q, ref, ans):
        return bool(parse_json(llm.invoke(JUDGE.format(question=q, reference=ref, answer=ans)).content,
                               {}).get("correct", False))

    systems = {"naive_rag": [], "docpilot_agent": []}
    for r in rows:
        hits_src = {h.source for h in pilot.store.search(r["question"], k=settings.top_k)}

        t0 = time.perf_counter()
        ans, _ = naive_rag(llm, pilot.store, r["question"], settings.top_k)
        systems["naive_rag"].append((r, ans, time.perf_counter() - t0, hits_src))

        t0 = time.perf_counter()
        out = pilot.ask(r["question"])
        systems["docpilot_agent"].append((r, out["answer"], time.perf_counter() - t0,
                                          {s["source"] for s in out["sources"]} or hits_src))
        print(f"done: {r['question'][:60]}")

    report = {}
    for name, results in systems.items():
        ans_rows = [x for x in results if x[0]["answer"] != "NOT_IN_DOCS"]
        unans_rows = [x for x in results if x[0]["answer"] == "NOT_IN_DOCS"]
        report[name] = {
            "retrieval_hit": round(sum(x[0]["source"] in x[3] for x in ans_rows) / max(len(ans_rows), 1), 3),
            "correct": round(sum(judge(x[0]["question"], x[0]["answer"], x[1]) for x in ans_rows)
                             / max(len(ans_rows), 1), 3),
            "abstained_ok": round(sum(bool(ABSTAIN.search(x[1])) for x in unans_rows)
                                  / max(len(unans_rows), 1), 3),
            "cited": round(sum(bool(re.search(r"\[\d+\]", x[1])) for x in ans_rows) / max(len(ans_rows), 1), 3),
            "avg_latency_s": round(sum(x[2] for x in results) / len(results), 2),
        }

    meta = {"llm": f"{settings.llm_provider}/{settings.llm_model}", "embeddings": settings.embedding_model,
            "top_k": settings.top_k, "questions": len(rows)}
    Path(args.out).write_text(json.dumps({"config": meta, "results": report}, indent=2))

    cols = ["retrieval_hit", "correct", "abstained_ok", "cited", "avg_latency_s"]
    print(f"\n{meta}\n")
    print(f"{'system':<16}" + "".join(f"{c:>15}" for c in cols))
    for name, m in report.items():
        print(f"{name:<16}" + "".join(f"{m[c]:>15}" for c in cols))


if __name__ == "__main__":
    main()
