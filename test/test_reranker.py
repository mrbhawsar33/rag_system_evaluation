from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker

retriever = HybridRetriever("data/processed/vehicle_chunks.json")

reranker = CrossEncoderReranker()

query = "What is the horsepower of the 2011 BMW 1 Series M?"

candidates = retriever.search(query, top_k=10)

reranked = reranker.rerank(query, candidates, top_k=3)

for doc, score in reranked:
    print("\nScore:", score)
    print("Title:", doc["title"])
    print("Doc ID:", doc["doc_id"])
    print(doc["text"])