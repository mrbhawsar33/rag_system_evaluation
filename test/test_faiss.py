from src.retrieval.faiss_retriever import FAISSRetriever

retriever = FAISSRetriever("data/processed/vehicle_chunks.json")

query = "What is the horsepower of the 2011 BMW 1 Series M?"

results = retriever.search(query, top_k=3)

for doc, score in results:
    print("\nScore:", score)
    print("Title:", doc["title"])
    print("Doc ID:", doc["doc_id"])
    print(doc["text"])

"""Expected output---
Score: 0.40112823 Title: The 2011 BMW 1 Series M Doc ID: the_2011_bmw_1_series_m_00000 The 2011 BMW 1 Series M is a compact coupe. It features a 6-cylinder premium unleaded engine producing 335 horsepower. The transmission is manual with rear drive. It has 2 doors. This model falls under the following categories: factory tuner, luxury, high-performance. Fuel efficiency is rated at 19 city MPG and 26 highway MPG. It has a popularity score of 3916. The MSRP for this vehicle is $46,135. Score: 0.49095273 Title: The 2011 BMW 1 Series Doc ID: the_2011_bmw_1_series_00003 The 2011 BMW 1 Series is a compact coupe. It features a 6-cylinder premium unleaded engine producing 230 horsepower. The transmission is manual with rear drive. It has 2 doors. This model falls under the following categories: luxury, performance. Fuel efficiency is rated at 18 city MPG and 28 highway MPG. It has a popularity score of 3916. The MSRP for this vehicle is $29,450. Score: 0.4937652 Title: The 2011 BMW 1 Series Doc ID: the_2011_bmw_1_series_00002 The 2011 BMW 1 Series is a compact coupe. It features a 6-cylinder premium unleaded engine producing 300 horsepower. The transmission is manual with rear drive. It has 2 doors. This model falls under the following categories: luxury, high-performance. Fuel efficiency is rated at 20 city MPG and 28 highway MPG. It has a popularity score of 3916. The MSRP for this vehicle is $36,350.

With L2 distance in FAISS, smaller score = more similar.
Score: 0.401  → closest vector
Score: 0.49   → further vectors
"""