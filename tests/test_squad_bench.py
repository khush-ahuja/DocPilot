import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

from squad_bench import CountingLLM, summarize  # noqa: E402


def _s(answerable, label, latency=1.0, calls=1, tokens=100):
    return {"answerable": answerable, "label": label, "latency_s": latency, "calls": calls, "tokens": tokens}


def test_summarize_separates_answerable_and_unanswerable():
    samples = [_s(True, "correct"), _s(True, "abstained"), _s(False, "abstained"), _s(False, "incorrect")]
    r = summarize(samples)
    assert r["answerable_correct"] == 0.5 and r["answerable_abstained"] == 0.5
    assert r["unanswerable_abstained"] == 0.5 and r["unanswerable_hallucinated"] == 0.5
    assert r["overall_accuracy"] == 0.5
    assert r["llm_calls_per_q"] == 1 and r["tokens_per_q"] == 100


def test_counting_llm_counts_calls_and_tokens():
    class Fake:
        def invoke(self, prompt):
            return SimpleNamespace(content="x", response_metadata={"token_usage": {"total_tokens": 42}})

    llm = CountingLLM(Fake())
    llm.invoke("a")
    llm.invoke("b")
    assert llm.snapshot() == (2, 84)
