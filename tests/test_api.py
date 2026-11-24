from conftest import ROOT, ScriptedLLM
from fastapi.testclient import TestClient

import api
from docpilot.config import Settings
from docpilot.graph import DocPilot


def test_ingest_and_ask(store):
    settings = Settings()
    api.app.dependency_overrides[api.get_pilot] = lambda: (DocPilot(ScriptedLLM(), store), settings)
    client = TestClient(api.app)

    assert client.get("/health").json()["status"] == "ok"

    path = ROOT / "sample_docs" / "product_spec_nx200.md"
    r = client.post("/ingest", files=[("files", (path.name, path.read_bytes(), "text/markdown"))])
    assert r.status_code == 200 and r.json()["ingested"]["product_spec_nx200.md"] > 0

    r = client.post("/ask", json={"question": "What payload capacity does the NX-200 robot have?"})
    body = r.json()
    assert r.status_code == 200 and "[1]" in body["answer"]
    assert body["sources"][0]["source"] == "product_spec_nx200.md"

    assert client.post("/ingest", files=[("files", ("x.exe", b"bin", "application/octet-stream"))]).status_code == 400
    assert client.post("/ask", json={"question": "  "}).status_code == 400
    api.app.dependency_overrides.clear()
