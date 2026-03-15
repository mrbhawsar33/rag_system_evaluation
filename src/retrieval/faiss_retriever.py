
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import src.config.env

class FAISSRetriever:

    def __init__(
        self,
        json_path,
        embedding_path="data/processed/embeddings.npy",
        index_path="data/processed/faiss.index",
        model_name="all-MiniLM-L6-v2"
    ):

        with open(json_path, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

        self.texts = [doc["text"] for doc in self.docs]

        # embedding model
        self.model = SentenceTransformer(model_name)

        # ---------- Load cached embeddings ----------
        if os.path.exists(embedding_path):

            print("Loading cached embeddings...")

            self.embeddings = np.load(embedding_path)

        else:

            print("Computing embeddings...")

            embeddings = self.model.encode(
                self.texts,
                show_progress_bar=True
            )

            self.embeddings = np.array(embeddings).astype("float32")

            np.save(embedding_path, self.embeddings)

            print("Embeddings saved.")

        # ---------- Load FAISS index ----------
        if os.path.exists(index_path):

            print("Loading FAISS index...")

            self.index = faiss.read_index(index_path)

        else:

            print("Building FAISS index...")

            dim = self.embeddings.shape[1]

            self.index = faiss.IndexFlatL2(dim)

            self.index.add(self.embeddings)

            faiss.write_index(self.index, index_path)

            print("FAISS index saved.")

    def search(self, query, top_k=5):

        query_embedding = self.model.encode([query]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []

        for i, idx in enumerate(indices[0]):
            results.append((self.docs[idx], distances[0][i]))

        return results