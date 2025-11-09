from typing import List, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BM25Retriever:
    """BM25 检索器：仅在传入的 kb_list（本地人设列表）上建立索引并检索。"""

    def __init__(self, kb_list: List[str]):
        print(f"BM25Retriever initializing with {len(kb_list)} personas...")
        self.kb_list: List[str] = [p.strip() for p in kb_list if str(p).strip()]
        tokenized_corpus = [doc.split(" ") for doc in self.kb_list]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, int]]:
        tokenized_query = query.split(" ")
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(doc_scores)[-top_k:][::-1]
        results: List[Tuple[str, int]] = []
        for idx in top_k_indices:
            original_index_1based = int(idx) + 1
            results.append((self.kb_list[int(idx)], original_index_1based))
        return results


class TFIDFRetriever:
    """TF-IDF 检索器：仅在传入的 kb_list（本地人设列表）上建立向量并检索。"""

    def __init__(self, kb_list: List[str]):
        print(f"TFIDFRetriever initializing with {len(kb_list)} personas...")
        self.kb_list: List[str] = [p.strip() for p in kb_list if str(p).strip()]
        self.vectorizer = TfidfVectorizer().fit(self.kb_list)
        self.kb_vecs = self.vectorizer.transform(self.kb_list)

    def retrieve(self, query: str, top_k: int) -> List[Tuple[str, int]]:
        if not query:
            return []
        try:
            question_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(question_vec, self.kb_vecs)[0]
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            results: List[Tuple[str, int]] = []
            for idx in top_k_indices:
                original_index_1based = int(idx) + 1
                results.append((self.kb_list[int(idx)], original_index_1based))
            return results
        except Exception as e:
            print(f"TF-IDF retrieve error: {e}")
            return []


class Retriever:
    """统一检索接口，支持 bm25 / tfidf 引擎。"""

    def __init__(self, kb_list: List[str], engine: str = "bm25"):
        self.engine = engine
        if engine == "bm25":
            self.retriever = BM25Retriever(kb_list=kb_list)
        elif engine == "tfidf":
            self.retriever = TFIDFRetriever(kb_list=kb_list)
        else:
            raise NotImplementedError(f"Engine {engine} is not implemented. Please use 'bm25' or 'tfidf'.")

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[str, int]]:
        return self.retriever.retrieve(query, top_k)

