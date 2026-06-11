"""
RAG module for Vietnamese BF+RAG experiments.

Components:
  - retriever.py      FAISS-based dense retriever (multilingual-MiniLM)
  - knowledge_base.py Vietnamese Wikipedia index builder
  - rag_pipeline.py   End-to-end retrieval + prompt augmentation
"""

from .retriever import ViRetriever
from .knowledge_base import build_vi_wiki_index, load_vi_wiki_index
from .rag_pipeline import RAGPipeline

__all__ = [
    "ViRetriever",
    "build_vi_wiki_index",
    "load_vi_wiki_index",
    "RAGPipeline",
]
