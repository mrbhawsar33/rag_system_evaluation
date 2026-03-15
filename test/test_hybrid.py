from src.retrieval.hybrid_retriever import HybridRetriever

retriever = HybridRetriever("data/processed/vehicle_chunks.json")

query = "What is the horsepower of the 2011 BMW 1 Series M?"

results = retriever.search(query, top_k=3)

for doc, score in results:
    print("\nScore:", score)
    print("Title:", doc["title"])
    print("Doc ID:", doc["doc_id"])
    print(doc["text"])

"""
Expected output (truncated for brevity) WITH cross-encoder reranking:
Score: 10.328224 Title: The 2011 BMW 1 Series M Doc ID: the_2011_bmw_1_series_m_00000 The 2011 BMW 1 Series M is a compact coupe. It features a 6-cylinder premium unleaded engine producing 335 horsepower. The transmission is manual with rear drive. It has 2 doors. This model falls under the following categories: factory tuner, luxury, high-performance. Fuel efficiency is rated at 19 city MPG and 26 highway MPG. It has a popularity score of 3916. The MSRP for this vehicle is $46,135. Score: 8.798607...
"""