from docpilot.ingest import Chunk, split_text


def test_split_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 60 for i in range(10))
    chunks = split_text(text, chunk_size=400, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 400 + 60 + 2 for c in chunks)
    # overlap: the start of each chunk repeats words from the end of the previous one
    assert chunks[1].split()[0] in chunks[0]


def test_split_hard_cuts_long_paragraph():
    chunks = split_text("x " * 2000, chunk_size=300, overlap=0)
    assert all(len(c) <= 300 for c in chunks)


def test_store_finds_right_document(store):
    hits = store.search("paid annual leave days per year", k=3)
    assert hits[0].source == "employee_handbook.md"
    assert "24 days" in " ".join(h.text for h in hits)


def test_sources_and_reingest_is_idempotent(store):
    before = store.count()
    assert store.sources() == ["employee_handbook.md", "it_security_policy.md", "product_spec_nx200.md"]
    store.add([Chunk(text="dup", source="employee_handbook.md", page=None, chunk_index=0)])
    assert store.count() == before  # same id -> overwritten, not duplicated
    store.delete_source("it_security_policy.md")
    assert "it_security_policy.md" not in store.sources()
