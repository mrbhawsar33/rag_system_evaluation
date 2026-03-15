from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline(
    json_path="data/processed/vehicle_chunks.json",
    retriever_type="hybrid",
    llm_model="mistral",
    retrieval_top_k=10,
    final_top_k=3
)

query = "What is the horsepower of the 2011 BMW 1 Series M?"

result = pipeline.run(query)

print("\n=== ANSWER ===\n")
print(result["answer"])

print("\n=== CITATIONS ===\n")
print(result["citations"])

print("\n=== CONTEXT ===\n")
print(result["context"])