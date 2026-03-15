from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever("data/processed/vehicle_chunks.json")

query = "What is the horsepower of the 2011 BMW 1 Series M?"

results = retriever.search(query, top_k=3)

for doc, score in results:
    print("\nScore:", score)
    print("Title:", doc["title"])
    print("Doc ID:", doc["doc_id"])
    print(doc["text"])