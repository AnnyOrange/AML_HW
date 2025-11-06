from typing import List

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25 本地检索器：仅在传入的本地人设列表上建立索引并检索。"""

    def __init__(self, local_persona_facts: List[str]):
        self.personas: List[str] = [p.strip() for p in local_persona_facts if str(p).strip()]
        tokenized = [doc.split() for doc in self.personas]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve_top_k(self, query: str, k: int = 2) -> List[str]:
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.personas[i] for i in top_indices]

    def retrieve_with_scores(self, query: str, top_k: int = 2) -> List[tuple]:
        scores = self.bm25.get_scores(query.split())
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.personas[i], float(scores[i])) for i in top_indices]


