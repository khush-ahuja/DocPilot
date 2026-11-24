from conftest import ScriptedLLM

from docpilot.graph import NOT_FOUND, DocPilot


def test_answers_with_citation(store):
    out = DocPilot(ScriptedLLM(), store).ask("How many days of paid annual leave do employees get?")
    assert "[1]" in out["answer"]
    assert out["sources"] and out["sources"][0]["source"] == "employee_handbook.md"
    assert out["grounded"] and out["rewrites"] == 0
    assert out["trace"][0] == "route: retrieve"


def test_small_talk_skips_retrieval(store):
    llm = ScriptedLLM()
    out = DocPilot(llm, store).ask("hello there")
    assert out["route"] == "direct" and out["sources"] == []
    assert not any(c.startswith("You check which") for c in llm.calls)


def test_rewrites_then_recovers(store):
    # First search is graded useless, so the agent rewrites the query and searches again.
    llm = ScriptedLLM(rewrite_to="NX-200 robot payload capacity kg", reject_first_grade=True)
    out = DocPilot(llm, store).ask("What payload can the NX-200 robot carry?")
    assert out["rewrites"] == 1
    assert "rewrote query -> 'NX-200 robot payload capacity kg'" in out["trace"]
    assert out["sources"] and out["sources"][0]["source"] == "product_spec_nx200.md"
    assert "[1]" in out["answer"]


def test_gives_up_cleanly_when_nothing_relevant(store):
    out = DocPilot(ScriptedLLM(rewrite_to="quarterly stock ticker symbol"), store, max_rewrites=2) \
        .ask("What is the stock ticker symbol?")
    assert out["answer"] == NOT_FOUND
    assert out["rewrites"] == 2 and out["sources"] == []


def test_ungrounded_answer_is_regenerated_once(store):
    llm = ScriptedLLM(ungrounded_first=True)
    out = DocPilot(llm, store).ask("What payload can the NX-200 robot carry?")
    assert out["grounded"]
    assert sum(c.startswith("Answer the question") for c in llm.calls) == 2
    assert "not grounded, retrying" in out["trace"]
