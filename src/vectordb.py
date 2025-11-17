"""
vectordb.py — Stable Vector DB Wrapper for RAG
Compatible with LangChain 0.3+, langchain-chroma, langchain-huggingface.
Includes:
 - Chroma vector store
 - HuggingFace embeddings
 - CrossEncoder reranking
 - Safe auto persistence
"""

import os
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Local persistent DB folder
CHROMA_DIR = Path(__file__).resolve().parents[1] / "chroma_store"


class VectorDB:
    def __init__(self, collection_name: str, embedding_model_name: str):
        """Initialize vector database with embeddings + reranker."""

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name

        print(f" Loading embedding model: {embedding_model_name}")
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        print(" Loading CrossEncoder reranker...")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        CHROMA_DIR.mkdir(exist_ok=True)

        # Initialize new Chroma implementation from langchain-chroma
        self._store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_fn,
            persist_directory=str(CHROMA_DIR)
        )

        print(f" Vector DB initialized at: {CHROMA_DIR}")

    # ------------------------------------------------------
    # DOCUMENT SPLITTING
    # ------------------------------------------------------
    def _split_documents(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split documents into small, focused chunks."""

        chunks = []

        for doc in docs:
            text = doc["content"]
            sections = text.split("\n\n")

            for section in sections:
                section = section.strip()
                if len(section) < 50:
                    continue

                if len(section) > 800:
                    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                        model_name="gpt-4o-mini",
                        chunk_size=300,
                        chunk_overlap=50,
                    )
                    pieces = splitter.split_text(section)

                    for piece in pieces:
                        chunks.append({
                            "content": piece,
                            "metadata": doc["metadata"]
                        })
                else:
                    chunks.append({
                        "content": section,
                        "metadata": doc["metadata"]
                    })

        print(f"  Split into {len(chunks)} chunks")
        return chunks

    # ------------------------------------------------------
    # ADD DOCUMENTS
    # ------------------------------------------------------
    def add_documents(self, docs: List[Dict[str, Any]]):
        """Add documents into Chroma vector store."""
        print(f" Processing {len(docs)} documents...")

        chunks = self._split_documents(docs)
        texts = [c["content"] for c in chunks]
        metas = [c["metadata"] for c in chunks]

        self._store.add_texts(texts=texts, metadatas=metas)

        # Auto persist (correct for langchain-chroma)
        if hasattr(self._store, "_client") and hasattr(self._store._client, "persist"):
            self._store._client.persist()

        print(f" Stored {len(texts)} chunks in collection '{self.collection_name}'")

    # ------------------------------------------------------
    # SEARCH / RETRIEVAL
    # ------------------------------------------------------
    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """Search using vector similarity + CrossEncoder reranking."""

        initial_k = min(n_results * 3, 20)

        try:
            results = self._store.similarity_search_with_score(
                query,
                k=initial_k
            )
        except Exception as e:
            print(f" Search error: {e}")
            return {"documents": [], "metadatas": [], "scores": []}

        if not results:
            return {"documents": [], "metadatas": [], "scores": []}

        docs, metas, initial_scores = [], [], []

        for doc, dist in results:
            docs.append(doc.page_content)
            metas.append(doc.metadata)
            # convert L2 → cosine similarity
            similarity = max(0, 1 - (dist ** 2) / 2)
            initial_scores.append(similarity)

        if docs:
            print(f" Initial retrieval: {len(docs)} docs — reranking...")

            pairs = [(query, d) for d in docs]
            rerank_scores = self.cross_encoder.predict(pairs)

            # Normalize 0–1
            if len(rerank_scores) > 1:
                mn, mx = float(np.min(rerank_scores)), float(np.max(rerank_scores))
                rg = mx - mn
                if rg > 0:
                    rerank_scores = [(s - mn) / rg for s in rerank_scores]
                else:
                    rerank_scores = [0.5] * len(rerank_scores)

            # Sort by best score
            order = np.argsort(rerank_scores)[::-1]
            top_idx = order[:n_results]

            docs = [docs[i] for i in top_idx]
            metas = [metas[i] for i in top_idx]
            scores = [rerank_scores[i] for i in top_idx]

            print(f" Reranked, top score = {scores[0]:.3f}")

        else:
            scores = []

        return {
            "documents": docs,
            "metadatas": metas,
            "scores": scores
        }

    # ------------------------------------------------------
    # CLEAR COLLECTION
    # ------------------------------------------------------
    def clear_collection(self):
        try:
            self._store.delete_collection()
            print(f" Cleared collection '{self.collection_name}'")
        except Exception as e:
            print(f" Error clearing collection: {e}")
