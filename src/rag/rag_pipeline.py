from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.reranking.cross_encoder_reranker import CrossEncoderReranker
from src.llm.ollama_client import OllamaClient
from src.prompts.prompts import PROMPT_V2


class RAGPipeline:
    def __init__(
        self,
        json_path: str,
        retriever_type: str = "hybrid",
        llm_model="mistral",
        prompt_template=None,
        retrieval_top_k: int = 10,
        final_top_k: int = 3,
    ):
        self.json_path = json_path
        self.retriever_type = retriever_type.lower()
        self.retrieval_top_k = retrieval_top_k
        self.final_top_k = final_top_k
        self.llm = OllamaClient(
            model_name=llm_model,
            prompt_template=prompt_template
        )

        if self.retriever_type == "bm25":
            self.retriever = BM25Retriever(json_path)
        elif self.retriever_type == "faiss":
            self.retriever = FAISSRetriever(json_path)
        elif self.retriever_type == "hybrid":
            self.retriever = HybridRetriever(json_path)
        else:
            raise ValueError(
                "retriever_type must be one of: 'bm25', 'faiss', 'hybrid'"
            )

        self.reranker = CrossEncoderReranker()

    def retrieve(self, query: str):
        return self.retriever.search(query, top_k=self.retrieval_top_k)

    def rerank(self, query: str, candidates):
        return self.reranker.rerank(query, candidates, top_k=self.final_top_k)

    def build_context(self, reranked_docs):
        context_parts = []

        for i, (doc, score) in enumerate(reranked_docs, start=1):
            context_parts.append(
                f"[{i}] {doc['title']} | doc_id={doc['doc_id']}\n{doc['text']}"
            )

        return "\n\n".join(context_parts)

    def run(self, query: str):

        candidates = self.retrieve(query)

        reranked_docs = self.rerank(query, candidates)

        context = self.build_context(reranked_docs)

        answer = self.llm.generate(query, context)

        return {
            "query": query,
            "retriever_type": self.retriever_type,
            "context": context,
            "answer": answer.answer,
            "citations": answer.citations,
            "reranked_docs": reranked_docs
        }