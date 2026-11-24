"""Corrective-RAG agent built with LangGraph.

    route ─┬─ direct ───────────────────────────────────────────────► END
           └─ retrieve ─► grade ─┬─ (relevant) ─► generate ─► check ─┬─► END
                  ▲              └─ (none) ─► rewrite ─┘             └─ (ungrounded, once) ─► generate
                  └──────────────────────────────┘

- route:    skip retrieval for small talk
- grade:    LLM keeps only passages that actually help
- rewrite:  if nothing is relevant, reformulate the query and search again (up to MAX_REWRITES)
- generate: answer only from kept passages, with [n] citations
- check:    LLM verifies the answer is grounded; one stricter retry if not
"""
import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from docpilot import prompts
from docpilot.llm import parse_json
from docpilot.store import Hit, VectorStore

NOT_FOUND = "I couldn't find information about that in the uploaded documents."


class RAGState(TypedDict, total=False):
    question: str
    query: str
    route: str
    hits: list[Hit]
    relevant: list[Hit]
    rewrites: int
    answer: str
    grounded: bool
    unsupported: str
    retries: int
    trace: list[str]


def _context(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.label()})\n{h.text}" for i, h in enumerate(hits, 1))


def build_graph(llm, store: VectorStore, top_k: int = 5, max_rewrites: int = 2):
    def ask(prompt: str) -> str:
        return llm.invoke(prompt).content

    def log(state, msg):
        return state.get("trace", []) + [msg]

    def route(state: RAGState):
        r = parse_json(ask(prompts.ROUTE.format(question=state["question"])), {}).get("route", "retrieve")
        r = r if r in ("retrieve", "direct") else "retrieve"
        return {"route": r, "query": state["question"], "rewrites": 0, "retries": 0,
                "trace": log(state, f"route: {r}")}

    def direct(state: RAGState):
        return {"answer": ask(prompts.DIRECT.format(question=state["question"])), "relevant": [],
                "grounded": True, "trace": log(state, "answered directly")}

    def retrieve(state: RAGState):
        hits = store.search(state["query"], k=top_k)
        return {"hits": hits, "trace": log(state, f"retrieved {len(hits)} for query: {state['query']!r}")}

    def grade(state: RAGState):
        hits = state["hits"]
        if not hits:
            return {"relevant": [], "trace": log(state, "graded: no passages")}
        passages = "\n\n".join(f"[{i}] {h.text[:1200]}" for i, h in enumerate(hits, 1))
        keep = parse_json(ask(prompts.GRADE.format(question=state["question"], passages=passages)), {})
        idx = [int(i) for i in keep.get("relevant", []) if str(i).isdigit() and 1 <= int(i) <= len(hits)]
        relevant = [hits[i - 1] for i in sorted(set(idx))]
        return {"relevant": relevant, "trace": log(state, f"graded: {len(relevant)}/{len(hits)} relevant")}

    def after_grade(state: RAGState):
        if state["relevant"]:
            return "generate"
        return "rewrite" if state["rewrites"] < max_rewrites else "generate"

    def rewrite(state: RAGState):
        out = parse_json(ask(prompts.REWRITE.format(question=state["question"], query=state["query"])), {})
        q = (out.get("query") or state["question"]).strip()
        return {"query": q, "rewrites": state["rewrites"] + 1, "trace": log(state, f"rewrote query -> {q!r}")}

    def generate(state: RAGState):
        rel = state.get("relevant", [])
        if not rel:
            return {"answer": NOT_FOUND, "grounded": True, "trace": log(state, "no relevant context")}
        prompt = prompts.GENERATE.format(context=_context(rel), question=state["question"])
        if state.get("grounded") is False:
            prompt += prompts.STRICT_SUFFIX.format(unsupported=state.get("unsupported") or "unspecified")
        return {"answer": ask(prompt).strip(), "trace": log(state, "generated answer")}

    def check(state: RAGState):
        if state["answer"] == NOT_FOUND:
            return {"grounded": True}
        out = parse_json(ask(prompts.GROUNDED.format(context=_context(state["relevant"]),
                                                     answer=state["answer"])), {"grounded": True})
        g = bool(out.get("grounded", True))
        return {"grounded": g, "unsupported": out.get("unsupported", ""),
                "retries": state["retries"] + (0 if g else 1),
                "trace": log(state, "grounded" if g else "not grounded, retrying")}

    def after_check(state: RAGState):
        return END if state["grounded"] or state["retries"] > 1 else "generate"

    g = StateGraph(RAGState)
    for name, fn in [("route", route), ("direct", direct), ("retrieve", retrieve), ("grade", grade),
                     ("rewrite", rewrite), ("generate", generate), ("check", check)]:
        g.add_node(name, fn)
    g.add_edge(START, "route")
    g.add_conditional_edges("route", lambda s: s["route"], {"direct": "direct", "retrieve": "retrieve"})
    g.add_edge("direct", END)
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, {END: END, "generate": "generate"})
    return g.compile()


class DocPilot:
    """Convenience wrapper: ask a question, get answer + cited sources + trace."""

    def __init__(self, llm, store: VectorStore, top_k: int = 5, max_rewrites: int = 2):
        self.store = store
        self.graph = build_graph(llm, store, top_k, max_rewrites)

    def ask(self, question: str) -> dict:
        t0 = time.perf_counter()
        s = self.graph.invoke({"question": question, "trace": []})
        rel = s.get("relevant", [])
        return {
            "answer": s["answer"],
            "sources": [{"id": i, "source": h.source, "page": h.page, "score": round(h.score, 4),
                         "text": h.text} for i, h in enumerate(rel, 1)],
            "grounded": s.get("grounded", True),
            "rewrites": s.get("rewrites", 0),
            "route": s.get("route"),
            "trace": s.get("trace", []),
            "latency_s": round(time.perf_counter() - t0, 2),
        }
