import json
import time
import csv

from src.rag.rag_pipeline import RAGPipeline
from src.prompts.prompts import PROMPT_V1, PROMPT_V2

class ExperimentRunner:

    def __init__(self, dataset_path, kb_path, output_path):

        self.dataset_path = dataset_path
        self.kb_path = kb_path
        self.output_path = output_path

        with open(dataset_path, "r") as f:
            self.questions = json.load(f)

        self.retrievers = ["bm25", "faiss", "hybrid"]
        self.llms = ["mistral", "qwen3:4b"]
        self.prompts = {
            "v1": PROMPT_V1,
            "v2": PROMPT_V2
        }

    def run(self):

        results = []

        for retriever in self.retrievers:
            for llm in self.llms:
                for prompt_name, prompt in self.prompts.items():

                    print(f"\nRunning configuration: {retriever} + {llm} + {prompt_name}\n")

                    pipeline = RAGPipeline(
                        json_path=self.kb_path,
                        retriever_type=retriever,
                        llm_model=llm,
                        prompt_template=prompt,
                        retrieval_top_k=10,
                        final_top_k=3
                    )

                    for item in self.questions:

                        qid = item["id"]
                        question = item["question"]
                        ground_truth = item["ground_truth"]

                        start = time.time()

                        output = pipeline.run(question)

                        latency = time.time() - start

                        results.append({
                            "question_id": qid,
                            "question": question,
                            "ground_truth": ground_truth,
                            "retriever": retriever,
                            "llm": llm,
                            "answer": output["answer"],
                            "citations": output["citations"],
                            "latency": latency
                        })

                        print(f"Q{qid} done")

        self.save_results(results)

    def save_results(self, results):

        keys = results[0].keys()

        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

        print("\nResults saved to:", self.output_path)