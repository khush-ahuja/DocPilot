"""Settings, read from environment variables (or a .env file)."""
import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class Settings:
    # LLM: "groq" (free tier), "openai", or "ollama" (fully local)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "groq"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama-3.1-8b-instant"))
    temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0")))

    # Embeddings (FastEmbed runs locally, no API key)
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"))

    # Qdrant: local folder by default; set QDRANT_URL to use a server or Qdrant Cloud
    qdrant_path: str = field(default_factory=lambda: os.getenv("QDRANT_PATH", "qdrant_data"))
    qdrant_url: str | None = field(default_factory=lambda: os.getenv("QDRANT_URL"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "docpilot"))

    # Retrieval and agent behaviour
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "120")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    max_rewrites: int = field(default_factory=lambda: int(os.getenv("MAX_REWRITES", "2")))
