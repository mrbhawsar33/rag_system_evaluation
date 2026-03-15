"""
Reciprocal Rank Fusion (RRF) score = 1 / (k + rank)

Returns:
    _type_: _description_
"""
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_retriever import FAISSRetriever


class HybridRetriever:

    def __init__(self, json_path, rrf_k=60):

        self.bm25 = BM25Retriever(json_path)
        self.faiss = FAISSRetriever(json_path)

        self.rrf_k = rrf_k

    def _rrf_score(self, rank):
        return 1 / (self.rrf_k + rank)

    def search(self, query, top_k=5):

        candidate_k = top_k * 5 # reranker needs a larger candidate pool to choose from.

        bm25_results = self.bm25.search(query, top_k=candidate_k)
        faiss_results = self.faiss.search(query, top_k=candidate_k)

        scores = {}
        docs = {}

        # BM25 ranking
        for rank, (doc, _) in enumerate(bm25_results, start=1):

            doc_id = doc["doc_id"]

            scores[doc_id] = scores.get(doc_id, 0) + self._rrf_score(rank)
            docs[doc_id] = doc

        # FAISS ranking
        for rank, (doc, _) in enumerate(faiss_results, start=1):

            doc_id = doc["doc_id"]

            scores[doc_id] = scores.get(doc_id, 0) + self._rrf_score(rank)
            docs[doc_id] = doc

        ranked_docs = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for doc_id, score in ranked_docs[:top_k]:
            results.append((docs[doc_id], score))

        return results