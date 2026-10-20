from __future__ import annotations

import bm25s

from .schema import Passage


class BM25Retriever:
    def __init__(self, passages: tuple[Passage, ...]):
        if not passages or len({p.id for p in passages}) != len(passages):
            raise ValueError("BM25 requires a nonempty unique corpus")
        self.passages = tuple(sorted(passages, key=lambda p: p.id))
        self.index = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
        tokens = bm25s.tokenize(
            [p.title + " " + p.text for p in self.passages], stopwords="en", show_progress=False
        )
        self.index.index(tokens, show_progress=False)

    def search(self, query: str, k: int, exclude: set[str]) -> tuple[Passage, ...]:
        if k < 1:
            raise ValueError("top-k must be positive")
        if not query.strip():
            return ()
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        ids, scores = self.index.retrieve(tokens, k=len(self.passages), show_progress=False)
        ranked = sorted(
            (
                (self.passages[int(i)], float(score))
                for i, score in zip(ids[0], scores[0], strict=True)
            ),
            key=lambda item: (-item[1], item[0].id),
        )
        return tuple(p for p, score in ranked if p.id not in exclude and score > 0)[:k]
