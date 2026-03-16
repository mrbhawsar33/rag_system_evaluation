# Evaluating RAG Systems in Vehicle Knowledge QA

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) evaluation framework** for answering factual questions about vehicle specifications. The system builds a knowledge base from a vehicle dataset and evaluates multiple **retriever + LLM combinations** to measure retrieval effectiveness, answer accuracy, and hallucination behavior.

The project emphasizes **measurement and evaluation** rather than just generation. It compares different retrieval strategies and local language models to understand their performance trade-offs.

---

## Features

* **Fact-grounded automotive knowledge base**
* **Paragraph-level chunking** (one chunk per vehicle)
* **Sparse retrieval (BM25)**
* **Dense retrieval (FAISS + sentence-transformers)**
* **Hybrid retrieval (Reciprocal Rank Fusion)**
* **Cross-encoder reranking**
* **Local LLM inference using Ollama**
* **Prompt versioning**
* **Structured outputs with citations**
* **Experiment runner for automated benchmarking**

---

## System Architecture

```
User Query
   │
   ▼
Retriever Layer
 ├── BM25
 ├── Dense (FAISS)
 └── Hybrid (RRF)
   │
   ▼
Cross-Encoder Reranker
   │
   ▼
Top-K Context
   │
   ▼
LLM Layer (Ollama)
 ├── Mistral
 └── Qwen
   │
   ▼
Structured Answer + Citations
   │
   ▼
Evaluation Framework
```

---

## Project Structure

```
rag-vehicle-evaluation/
│
├── data/
│   ├── raw/
│   │   └── vehicle_knowledge_base.txt
│   │
│   └── processed/
│       ├── vehicle_chunks.json
│       ├── embeddings.npy
│       └── faiss.index
│
├── src/
│   │
│   ├── config/
│   │   └── env.py
│   │
│   ├── data/
│   │   └── chunk_dataset.py
│   │
│   ├── retrieval/
│   │   ├── bm25_retriever.py
│   │   ├── faiss_retriever.py
│   │   └── hybrid_retriever.py
│   │
│   ├── reranking/
│   │   └── cross_encoder_reranker.py
│   │
│   ├── rag/
│   │   └── rag_pipeline.py
│   │
│   ├── llm/
│   │   ├── ollama_client.py
│   │   └── structured_output.py
│   │
│   ├── prompts/
│   │   └── prompts.py
│   │
│   └── evaluation/
│       ├── question_set.json
│       └── experiment_runner.py
│
├── test/
│   ├── test_bm25.py
│   ├── test_faiss.py
│   ├── test_hybrid.py
│   ├── test_reranker.py
│   ├── test_rag_pipeline.py
│   └── run_experiments.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup Instructions

### 1. Clone the Repository

```
git clone <your-repo-url>
cd rag-vehicle-evaluation
```

---

### 2. Create Virtual Environment

#### Linux / Mac

```
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```
python -m venv venv
venv\Scripts\activate
```

---

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

### 4. Install and Start Ollama

Download Ollama:

```
https://ollama.com
```

Pull required models:

```
ollama pull mistral
ollama pull qwen3:4b
```

Verify installation:

```
ollama list
```

---

## Running the System

### Test Individual Components

BM25 Retrieval

```
python -m test.test_bm25
```

Dense Retrieval (FAISS)

```
python -m test.test_faiss
```

Hybrid Retrieval

```
python -m test.test_hybrid
```

Cross-Encoder Reranker

```
python -m test.test_reranker
```

---

### Run Full RAG Pipeline

```
python -m test.test_rag_pipeline
```

---

### Run Evaluation Experiments

Runs all combinations of retrievers and LLMs across the evaluation dataset.

```
python -m test.run_experiments
```

This will generate an output file such as:

```
evaluation_results.csv
```

containing answers, citations, and latency for each configuration.

---

## Retrieval Configurations

The experiment runner evaluates the following setups:

```
BM25 + Mistral
Dense (FAISS) + Mistral
Hybrid (RRF) + Mistral
BM25 + Qwen
Dense (FAISS) + Qwen
Hybrid (RRF) + Qwen
```

---

## Evaluation Dataset

The evaluation set contains **30 standardized questions** covering:

* Horsepower
* MSRP
* Fuel efficiency
* Vehicle type
* Categories
* Reverse lookup questions

These questions are stored in:

```
src/evaluation/question_set.json
```

---

## Future Improvements

Planned additions include:

* RAGAS evaluation metrics
* Retrieval accuracy metrics
* Hallucination detection
* Latency and token monitoring
* Experiment dashboard
* Observability with Langfuse

---

## License

This project is developed as part of an academic capstone on **RAG system evaluation**.
