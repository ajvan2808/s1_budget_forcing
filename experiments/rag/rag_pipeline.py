"""
End-to-end RAG pipeline: retrieve passages → augment prompt.

Vietnamese prompt template:
    [Ngữ cảnh tham khảo]
    {passage_1}
    {passage_2}
    {passage_3}

    Câu hỏi: {question}

Usage:
    pipeline = RAGPipeline.from_index("experiments/data/vi_wiki_index/")
    augmented_prompt = pipeline.augment(question, tokenizer)
    token_count = pipeline.last_retrieved_tokens  # for logging
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .retriever import ViRetriever, DEFAULT_TOP_K

# Vietnamese context header
CONTEXT_HEADER = "[Ngữ cảnh tham khảo]"
QUESTION_PREFIX = "Câu hỏi:"


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for Vietnamese tasks.

    Wraps ViRetriever to produce augmented prompt strings suitable for
    direct input to BudgetForcingDecoder or greedy decoding.
    """

    def __init__(
        self,
        retriever: ViRetriever,
        top_k: int = DEFAULT_TOP_K,
        max_passage_tokens: int = 128,
    ):
        self.retriever = retriever
        self.top_k = top_k
        self.max_passage_tokens = max_passage_tokens
        self._last_passages: List[str] = []
        self._last_retrieved_tokens: int = 0

    @classmethod
    def from_index(
        cls,
        index_dir: str | Path,
        top_k: int = DEFAULT_TOP_K,
        max_passage_tokens: int = 128,
    ) -> "RAGPipeline":
        """Load retriever from disk and return a ready pipeline."""
        retriever = ViRetriever.from_index(index_dir)
        return cls(retriever, top_k=top_k, max_passage_tokens=max_passage_tokens)

    @classmethod
    def disabled(cls) -> "RAGPipeline":
        """
        Return a no-op pipeline for BF_only condition.
        augment() returns the question unchanged.
        """
        return _DisabledRAGPipeline()

    def _truncate_passage(self, passage: str) -> str:
        """Truncate passage to max_passage_tokens (whitespace approximation)."""
        tokens = passage.split()
        if len(tokens) <= self.max_passage_tokens:
            return passage
        return " ".join(tokens[: self.max_passage_tokens]) + "..."

    def retrieve_passages(self, query: str) -> List[str]:
        """Retrieve and truncate passages for a query."""
        raw = self.retriever.retrieve_text(query, top_k=self.top_k)
        passages = [self._truncate_passage(p) for p in raw]
        self._last_passages = passages
        self._last_retrieved_tokens = sum(len(p.split()) for p in passages)
        return passages

    def build_context_block(self, passages: List[str]) -> str:
        """Format retrieved passages into a context block."""
        if not passages:
            return ""
        lines = [CONTEXT_HEADER]
        for i, p in enumerate(passages, 1):
            lines.append(f"({i}) {p}")
        return "\n".join(lines)

    def augment(self, question: str, tokenizer=None) -> str:
        """
        Retrieve passages and return the augmented question string.

        The returned string is the full user-turn content (not yet wrapped
        in chat template). Pass to format_prompt_vi() in run_eval_vi.py.

        Args:
            question:  Original question text
            tokenizer: Unused (kept for API compatibility)

        Returns:
            Augmented question with context block prepended
        """
        passages = self.retrieve_passages(question)
        if not passages:
            return question

        context_block = self.build_context_block(passages)
        augmented = f"{context_block}\n\n{QUESTION_PREFIX} {question}"
        return augmented

    @property
    def last_retrieved_tokens(self) -> int:
        """Approximate token count of last retrieved passages (for logging)."""
        return self._last_retrieved_tokens

    @property
    def last_passages(self) -> List[str]:
        """Passages from the last retrieve() call."""
        return list(self._last_passages)

    @property
    def enabled(self) -> bool:
        return True


class _DisabledRAGPipeline(RAGPipeline):
    """
    No-op pipeline for BF_only condition.
    Returns the question unchanged with zero retrieved tokens.
    """

    def __init__(self):
        # Don't call super().__init__() — no retriever needed
        self._last_passages = []
        self._last_retrieved_tokens = 0
        self.top_k = 0
        self.max_passage_tokens = 0

    def retrieve_passages(self, query: str) -> List[str]:
        return []

    def augment(self, question: str, tokenizer=None) -> str:
        return question

    @property
    def enabled(self) -> bool:
        return False
