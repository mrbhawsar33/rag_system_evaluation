import pandas as pd


def analyze(csv_path):
    df = pd.read_csv(csv_path)

    summary = df.groupby(["retriever", "llm", "prompt"]).agg({
        "recall_at_k": "mean",
        "mrr": "mean",
        "hallucination": "mean",
        "latency": "mean"
    }).reset_index()

    print("\n=== Aggregated Results ===\n")
    print(summary)

    return summary


if __name__ == "__main__":
    analyze("./data/output/evaluation_results.csv")