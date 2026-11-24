"""DocPilot: an agentic RAG assistant built with LangGraph and Qdrant."""
from docpilot.config import Settings


def create_pilot(settings: Settings | None = None):
    """Builds the default app: FastEmbed + Qdrant + the configured LLM."""
    from docpilot.embeddings import FastEmbedEmbedder
    from docpilot.graph import DocPilot
    from docpilot.llm import get_llm
    from docpilot.store import VectorStore

    settings = settings or Settings()
    store = VectorStore.from_settings(settings, FastEmbedEmbedder(settings.embedding_model))
    return DocPilot(get_llm(settings), store, settings.top_k, settings.max_rewrites), settings
