import json
from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.corpus = [doc["text"] for doc in self.docs]

        # simple tokenization
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=5):

        tokenized_query = query.lower().split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            zip(self.docs, scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return ranked