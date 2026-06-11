"""
Dense retriever using FAISS + multilingual sentence embeddings.

Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  - 50 languages including Vietnamese
  - 384-dim embeddings, ~120 MB download
  - CPU-feasible

Usage:
    retriever = ViRetriever.from_index("data/vi_wiki_index/")
    passages = retriever.retrieve("diện tích của Việt Nam là bao nhiêu?", top_k=3)
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict

import numpy as np


EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_TOP_K = 3
MAX_TOKENS_PER_PASSAGE = 128


class ViRetriever:
    """
    FAISS flat-L2 retriever with multilingual-MiniLM embeddings.

    Attributes:
        index:    faiss.IndexFlatL2 loaded from disk
        passages: list of passage strings (parallel to index rows)
        model:    SentenceTransformer encoder
    """

    def __init__(self, index, passages: List[str], model):
        self.index = index
        self.passages = passages
        self.model = model

    @classmethod
    def from_index(cls, index_dir: str | Path) -> "ViRetriever":
        """Load a pre-built FAISS index from disk."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Install faiss-cpu and sentence-transformers: "
                "pip install faiss-cpu sentence-transformers"
            ) from e

        index_dir = Path(index_dir)
        if not index_dir.exists():
            raise FileNotFoundError(
                f"Index directory not found: {index_dir}\n"
                "Run experiments/rag/knowledge_base.py to build it first."
            )

        index = faiss.read_index(str(index_dir / "index.faiss"))

        with open(index_dir / "passages.pkl", "rb") as f:
            passages = pickle.load(f)

        model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"[ViRetriever] Loaded index: {len(passages)} passages from {index_dir}")
        return cls(index, passages, model)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized embeddings."""
        import faiss
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict]:
        """
        Retrieve top-k passages for a query.

        Returns:
            List of dicts: [{"passage": str, "score": float, "rank": int}, ...]
        """
        q_emb = self.encode([query])  # (1, 384)
        distances, indices = self.index.search(q_emb, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0 or idx >= len(self.passages):
                continue
            results.append({
                "passage": self.passages[idx],
                "score": float(dist),
                "rank": rank + 1,
            })
        return results

    def retrieve_text(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
        """Return just the passage strings (convenience method)."""
        return [r["passage"] for r in self.retrieve(query, top_k=top_k)]

    @property
    def num_passages(self) -> int:
        return len(self.passages)
