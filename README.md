# Evaluating RAG Systems in Vehicle Knowledge QA

## Overview

This project evaluates multiple Retrieval-Augmented Generation (RAG) system configurations on an automotive knowledge base. The goal is to measure how different retrieval strategies, LLMs, and prompts impact retrieval effectiveness, answer accuracy, and hallucination behavior.

The system is fully custom-built without external RAG frameworks and includes an end-to-end evaluation pipeline along with an interactive dashboard.

---

## Objectives

- Build a fact-grounded automotive QA system
- Compare multiple RAG configurations
- Quantify hallucinations and answer quality
- Provide a reproducible evaluation framework

---

## Dataset

The dataset consists of automotive vehicle specifications converted into structured text chunks.

Each chunk contains:

- title
- text
- doc_id

Stored in:
data/processed/vehicle_chunks.json

---

## RAG Pipeline

### Retrieval Methods

- BM25 (Sparse Retrieval)
- FAISS (Dense Retrieval)
- Hybrid (BM25 + FAISS)

### Reranking

- Cross-encoder reranker for improved precision

### LLMs (Local via Ollama)

- Mistral
- Qwen3:4b

### Prompting

- Prompt Versioning:
  - v1
  - v2 (improved hallucination control)

---

## Question Set

The question set is generated programmatically from the dataset.

Includes:

- Factual questions
- Reverse lookup questions
- Filtering queries
- Hallucination traps (NOT_FOUND)

Each question contains:

- question
- ground_truth
- type
- reference_doc_id

---

## Evaluation Metrics

### Retrieval Metrics

- Recall@K
- MRR (Mean Reciprocal Rank)

### LLM Evaluation

- Hallucination Rate (custom logic)
- RAGAS Metrics:
  - Faithfulness
  - Answer Relevancy

---

## Experiment Configurations

The system evaluates combinations of:

- Retriever:
  - BM25
  - FAISS
  - Hybrid

- LLM:
  - Mistral
  - Qwen3

- Prompt:
  - v1
  - v2

Results are stored in:
data/output/evaluation_results.csv

---

## Dashboard

An interactive dashboard is built using Streamlit.

Features:

- Filter by retriever, LLM, and prompt
- Metric comparison charts
- Aggregated performance metrics
- Automatic best configuration detection

Run dashboard:
streamlit run dashboard/app.py

---

## Key Findings

- BM25 outperformed FAISS in retrieval quality
- Hybrid performed similarly to BM25 without significant gains
- Mistral and Qwen showed similar accuracy
- Mistral was significantly faster
- Prompt v2 reduced hallucinations
- Retrieval quality had more impact than LLM choice

---

## Tech Stack

- Python
- FAISS
- BM25
- Sentence Transformers
- Ollama (Local LLMs)
- RAGAS
- Streamlit

---

## Setup Instructions

### 1. Clone Repository

git clone <your-repo-url>
cd <repo-name>

### 2. Create Virtual Environment

python -m venv venv

Activate:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Setup Environment Variables

Create a .env file:

HF_TOKEN=your_huggingface_token

### 5. Run Pipeline

python experiment_runner.py

### 6. Launch Dashboard

streamlit run dashboard/app.py

---

## Project Structure

```
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── vehicle_chunks.json
│   │   ├── embeddings.npy
│   │   └── faiss.index
│   └── output/
│       └── evaluation_results.csv
│
├── src/
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── faiss_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── reranker/
│   ├── llm/
│   ├── evaluation/
│   └── utils/
│
├── dashboard/
│   └── app.py
│
├── experiment_runner.py
├── question_set.json
├── requirements.txt
└── README.md
```

---

## Future Improvements

- Add CI-based evaluation gating
- Integrate observability tools (e.g., Langfuse)
- Improve hybrid retrieval weighting
- Expand question set for robustness

---

## Team

- Mayur
- Pratham
- Yash
- Mili

---

## License

For academic use.
