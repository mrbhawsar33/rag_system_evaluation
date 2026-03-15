# rag_system_evaluation
Evaluating RAG systems in vehicle knowledge QA

project_root
│
├── data
│   ├── raw
│   └── processed
│
src/

config/
   env.py

retrieval/
   bm25_retriever.py
   faiss_retriever.py
   hybrid_retriever.py

reranking/
   cross_encoder_reranker.py
│
├── test
│   ├── __init__.py
│   ├── test_bm25.py
│   └── test_faiss.py
│
└── venv
