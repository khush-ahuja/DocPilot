"""Thin wrapper around a Qdrant collection."""
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from docpilot.embeddings import Embedder
from docpilot.ingest import Chunk


@dataclass
class Hit:
    text: str
    source: str
    page: int | None
    score: float

    def label(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


class VectorStore:
    def __init__(self, client: QdrantClient, embedder: Embedder, collection: str = "docpilot"):
        self.client, self.embedder, self.collection = client, embedder, collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection, vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE))

    @classmethod
    def from_settings(cls, settings, embedder: Embedder):
        if settings.qdrant_url:
            client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
        else:
            client = QdrantClient(path=settings.qdrant_path)
        return cls(client, embedder, settings.collection)

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> int:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            vectors = self.embedder.embed_documents([c.text for c in batch])
            points = [
                PointStruct(
                    # Stable id: re-ingesting the same file overwrites instead of duplicating.
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.source}|{c.page}|{c.chunk_index}")),
                    vector=v,
                    payload={"text": c.text, "source": c.source, "page": c.page},
                )
                for c, v in zip(batch, vectors)
            ]
            self.client.upsert(self.collection, points=points)
        return len(chunks)

    def delete_source(self, source: str):
        self.client.delete(self.collection, points_selector=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value=source))]))

    def search(self, query: str, k: int = 5) -> list[Hit]:
        res = self.client.query_points(self.collection, query=self.embedder.embed_query(query), limit=k)
        return [Hit(p.payload["text"], p.payload["source"], p.payload.get("page"), p.score) for p in res.points]

    def count(self) -> int:
        return self.client.count(self.collection).count

    def sources(self) -> list[str]:
        seen, offset = set(), None
        while True:
            points, offset = self.client.scroll(self.collection, limit=256, offset=offset,
                                                with_payload=["source"], with_vectors=False)
            seen.update(p.payload["source"] for p in points)
            if offset is None:
                return sorted(seen)
