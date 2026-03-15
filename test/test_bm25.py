from src.retrieval.bm25_retriever import BM25Retriever      

retriever = BM25Retriever("data/processed/vehicle_chunks.json")

query = "What is the horsepower of the 2011 BMW 1 Series M?"

results = retriever.search(query, top_k=3)

for doc, score in results:
    print("\nScore:", score)
    print("Title:", doc["title"])
    print("Doc ID:", doc["doc_id"])
    print(doc["text"])

"""
Expected output---
Score: 30.15338347447875
Title: The 2011 BMW 1 Series
Doc ID: the_2011_bmw_1_series_00002
The 2011 BMW 1 Series is a compact coupe.
It features a 6-cylinder premium unleaded engine producing 300 horsepower.
The transmission is manual with rear drive.
It has 2 doors.
This model falls under the following categories: luxury, high-performance.
Fuel efficiency is rated at 20 city MPG and 28 highway MPG.
It has a popularity score of 3916.
The MSRP for this vehicle is $36,350.
"""