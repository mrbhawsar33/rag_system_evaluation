from sentence_transformers import CrossEncoder

import src.config.env

class CrossEncoderReranker:

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs, top_k=5):

        # docs = [(doc, score), (doc, score)]

        pairs = [(query, doc["text"]) for doc, _ in docs]

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []

        for (doc, _), score in ranked[:top_k]:
            results.append((doc, score))

        return results