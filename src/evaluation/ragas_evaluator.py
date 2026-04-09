import pandas as pd
from datasets import Dataset

from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_community.llms import Ollama
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
)
# Create Ollama LLM
ollama_llm = Ollama(model="qwen3:4b")  # or qwen

ragas_llm = LangchainLLMWrapper(ollama_llm)

class RAGASEvaluator:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df = df.head(20)
        # Fill NaNs in entire dataframe
        df = df.fillna("")

        # Force ALL relevant columns to string safely
        for col in ["question", "answer", "context", "ground_truth"]:
            df[col] = df[col].apply(lambda x: str(x) if not isinstance(x, str) else x)

        # Debug check (optional but useful)
        print("Sample types:")
        print(type(df["context"].iloc[0]), df["context"].iloc[0][:50])
        df = df.astype(str)
        data = {
            "question": df["question"].tolist(),
            "answer": df["answer"].tolist(),
            "contexts": [[str(c)] for c in df["context"].tolist()],  # DOUBLE SAFE
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
            ],
            llm=ragas_llm,
            embeddings=embeddings,
            batch_size=1
        )

        print("\nRAGAS Evaluation Results:\n")
        print(result)

        return result