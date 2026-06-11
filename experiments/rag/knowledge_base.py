"""
Build a FAISS index from Vietnamese Wikipedia.

Source:  wikimedia/wikipedia, config="20231101.vi"
  - ~1.5M articles, ~2GB download
  - Use --max_docs for smoke/dev (e.g. 10000)

Output layout:
    data/vi_wiki_index/
        index.faiss       FAISS FlatL2 index
        passages.pkl      List[str] parallel to index rows
        meta.json         Build metadata

Usage:
    # Full index (takes ~30-60 min on CPU)
    python experiments/rag/knowledge_base.py --output_dir experiments/data/vi_wiki_index

    # Smoke (fast, first 10k docs)
    python experiments/rag/knowledge_base.py \
        --output_dir experiments/data/vi_wiki_index_smoke \
        --max_docs 10000 \
        --chunk_size 128
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE_TOKENS = 128   # approximate, splits on whitespace
STRIDE = 0                # no overlap for simplicity


def tokenize_approx(text: str) -> List[str]:
    """Simple whitespace tokenizer — approximates token count for chunking."""
    return text.split()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_TOKENS) -> List[str]:
    """
    Split text into chunks of ≤chunk_size whitespace tokens.
    Returns at least one chunk even if text is empty.
    """
    tokens = tokenize_approx(text)
    if not tokens:
        return [""]
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def clean_wiki_text(text: str) -> str:
    """Basic cleanup of Wikipedia article text."""
    # Remove section headers
    text = re.sub(r"==+[^=]+=+", " ", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_vi_wiki_index(
    output_dir: str | Path,
    max_docs: int | None = None,
    chunk_size: int = CHUNK_SIZE_TOKENS,
    batch_size: int = 256,
) -> Path:
    """
    Download Vi Wikipedia, chunk articles, build FAISS index.

    Args:
        output_dir: directory to save index.faiss + passages.pkl + meta.json
        max_docs:   cap on number of Wikipedia articles to process (None = all)
        chunk_size: max whitespace-tokens per passage chunk
        batch_size: embedding batch size

    Returns:
        Path to output_dir
    """
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Install dependencies: pip install faiss-cpu sentence-transformers datasets"
        ) from e

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KB] Loading Vietnamese Wikipedia (20231101.vi)...")
    dataset = load_dataset(
        "wikimedia/wikipedia",
        "20231101.vi",
        split="train",
        trust_remote_code=True,
    )

    n_docs = min(max_docs, len(dataset)) if max_docs else len(dataset)
    print(f"[KB] Processing {n_docs} articles (chunk_size={chunk_size} tokens)...")

    passages: List[str] = []
    for i in range(n_docs):
        text = clean_wiki_text(dataset[i]["text"])
        if len(text) < 20:
            continue
        for chunk in chunk_text(text, chunk_size):
            passages.append(chunk)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{n_docs} articles → {len(passages)} passages")

    print(f"[KB] Total passages: {len(passages)}")

    print(f"[KB] Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("[KB] Encoding passages (this takes a while on CPU)...")
    all_embeddings = []
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(emb.astype("float32"))
        if (start // batch_size + 1) % 50 == 0:
            pct = 100 * (start + batch_size) / len(passages)
            print(f"  Encoded {min(start+batch_size, len(passages))}/{len(passages)} ({pct:.0f}%)")

    embeddings = np.vstack(all_embeddings)
    dim = embeddings.shape[1]  # 384 for MiniLM-L12

    print(f"[KB] Building FAISS FlatL2 index (dim={dim})...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(output_dir / "index.faiss"))
    with open(output_dir / "passages.pkl", "wb") as f:
        pickle.dump(passages, f)

    meta = {
        "source": "wikimedia/wikipedia/20231101.vi",
        "n_docs_processed": n_docs,
        "n_passages": len(passages),
        "chunk_size_tokens": chunk_size,
        "embed_model": EMBED_MODEL_NAME,
        "faiss_index_type": "FlatL2",
        "embedding_dim": dim,
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[KB] Index saved to: {output_dir}")
    print(f"[KB]   index.faiss:  {(output_dir/'index.faiss').stat().st_size/1e6:.1f} MB")
    print(f"[KB]   passages.pkl: {(output_dir/'passages.pkl').stat().st_size/1e6:.1f} MB")
    return output_dir


def load_vi_wiki_index(index_dir: str | Path):
    """
    Convenience loader — returns (index, passages) tuple.
    Prefer ViRetriever.from_index() for full retriever functionality.
    """
    import faiss, pickle
    index_dir = Path(index_dir)
    index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "passages.pkl", "rb") as f:
        passages = pickle.load(f)
    return index, passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Vietnamese Wikipedia FAISS index")
    parser.add_argument(
        "--output_dir",
        default="experiments/data/vi_wiki_index",
        help="Directory to save the built index",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="Max articles to process (default: all ~1.5M). Use 10000 for smoke.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=CHUNK_SIZE_TOKENS,
        help=f"Max tokens per passage chunk (default: {CHUNK_SIZE_TOKENS})",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Embedding batch size (default: 256)",
    )
    args = parser.parse_args()

    build_vi_wiki_index(
        output_dir=args.output_dir,
        max_docs=args.max_docs,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
    )
