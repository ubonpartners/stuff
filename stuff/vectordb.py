import json
from typing import List, Optional, Dict, Any

class SimpleVectorDB:
    def __init__(self, collection_name: str = "events"):
        # Deferred import
        import chromadb
        from chromadb.config import Settings

        self._chromadb = chromadb
        self._client = chromadb.Client(Settings(anonymized_telemetry=False))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def num_items(self):
        return self._collection.count()

    def add(
        self,
        uid: str,
        embedding: List[float],
        start_time: float,
        end_time: float,
        stream_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ):
        """Add a single embedding with metadata (timestamp, stream_id) and a dictionary payload."""
        self._collection.add(
            ids=[uid],
            embeddings=[embedding],
            metadatas=[{"start_time": start_time, "end_time": end_time, "stream_id": stream_id}],
            documents=[str(payload or {})]  # store as serialized string
        )

    def query(
        self,
        embedding: List[float],
        top_k: int = 5,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        stream_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Query nearest vectors with optional float timestamp and int stream_id filtering."""
        where: Dict[str, Any] = {}

        if start_time is not None:
            where["end_time"] = {"$gte": start_time}
        if end_time is not None:
            where["start_time"] = {"$lte": end_time}
        if stream_ids:
            where["stream_id"] = {"$in": stream_ids}

        # now WHERE is never an empty {} unless no filters were given
        # but Chroma still doesn’t like an empty dict:
        if not where:
            where = None

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "uid": results["ids"][0][i],
                "distance": results["distances"][0][i],
                "payload": eval(results["documents"][0][i]),  # decode string back to dict
                "metadata": results["metadatas"][0][i],
            })
        return matches

    def export_to_json(self, path: str):
        """Export all entries to a JSON file."""
        results = self._collection.get(include=["embeddings", "metadatas", "documents"])
        with open(path, 'w') as f:
            json.dump(results, f)

    def import_from_json(self, path: str):
        """Import entries from a JS ON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self._collection.add(
            ids=data["ids"],
            embeddings=data["embeddings"],
            metadatas=data["metadatas"],
            documents=data["documents"]
        )
