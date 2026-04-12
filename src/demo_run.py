from src.rag.rag_pipeline import RAGPipeline
from src.prompts.prompts import PROMPT_V1, PROMPT_V2

# Demo script to run the RAG pipeline with different configurations
QUESTION = "What is the horsepower of The 2011 BMW 1 Series M?"
#  "type": "fact"
# "reference_doc_id": "the_2011_bmw_1_series_m_00000"
# "ground_truth": "335 horsepower"

retrievers = ["bm25", "faiss", "hybrid"]
llms = ["mistral", "qwen3:4b"]
prompts = {
    "v1": PROMPT_V1,  
    "v2": PROMPT_V2 
}

for retriever in retrievers:
    for llm in llms:
        for prompt_name, prompt in prompts.items():

            print("\n" + "="*50)
            print(f"Retriever: {retriever} | LLM: {llm} | Prompt: {prompt_name}")

            pipeline = RAGPipeline(
                json_path="data/processed/vehicle_chunks.json",
                retriever_type=retriever,
                llm_model=llm,
                prompt_template=prompt,
                retrieval_top_k=5,   # smaller
                final_top_k=2        # smaller
            )

            output = pipeline.run(QUESTION)

            print("Answer:", output["answer"])
            print("Top Docs:", output["retrieved_docs"][:2])