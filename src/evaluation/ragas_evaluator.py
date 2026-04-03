import pandas as pd
from datasets import Dataset

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate


class RAGASEvaluator:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_data(self):
        df = pd.read_csv(self.csv_path)

        # convert to RAGAS format
        data = {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": df["context"].tolist(),  # IMPORTANT
            "ground_truth": df["ground_truth"].tolist(),
        }

        return Dataset.from_dict(data)

    def run(self):
        dataset = self.load_data()

        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy
            ]
        )

        print("\nRAGAS Evaluation Results:\n")
        print(result)

        return result