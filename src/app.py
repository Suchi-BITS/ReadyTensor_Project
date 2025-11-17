"""
app.py — RAG Console Application (LangChain 1.x, enhanced)
Loads .txt docs, builds embeddings, retrieves context, and answers with
OpenAI / Groq / Gemini fallback + reranking + token-based chunking.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment
load_dotenv()

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
#from langchain_google_genai import ChatGoogleGenerativeAI

# Local module
from vectordb import VectorDB

# Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def load_documents() -> List[Dict[str, Any]]:
    """Load all .txt files from data directory"""
    docs = []
    for f in DATA_DIR.glob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8")
            docs.append({"content": text, "metadata": {"filename": f.name}})
        except Exception as e:
            print(f"[app] Error reading {f.name}: {e}")
    return docs


class RAGApp:
    def __init__(self):
        print("\n Initializing RAG App (LangChain 1.x Enhanced)")
        self.vdb = VectorDB(CHROMA_COLLECTION_NAME, EMBEDDING_MODEL)
        self.llm = self._load_llm()
        self.prompt_template = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use ONLY the context below to answer the question.\n"
            "Be concise and specific - answer ONLY what is asked.\n"
            "If asked about a specific topic (like 'Deep Learning'), provide only information about that topic.\n"
            "Do not include information about other related topics unless specifically asked.\n"
            "If the context doesn't contain the answer, say 'I don't have enough information to answer that.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Provide a focused, concise answer:"
        )

    def _load_llm(self):
        """Load LLM with fallback priority: OpenAI -> Groq -> Gemini"""
        if OPENAI_API_KEY:
            try:
                print(f" Using OpenAI model: {OPENAI_MODEL}")
                return ChatOpenAI(model=OPENAI_MODEL, temperature=0)
            except Exception as e:
                print(f"⚠ OpenAI failed: {e}")
        
        if GROQ_API_KEY:
            try:
                print(f" Using Groq model: {GROQ_MODEL}")
                return ChatGroq(model=GROQ_MODEL, temperature=0)
            except Exception as e:
                print(f"⚠ Groq failed: {e}")
        
        '''if GOOGLE_API_KEY:
            try:
                print(f" Using Gemini model: {GOOGLE_MODEL}")
                return ChatGoogleGenerativeAI(model=GOOGLE_MODEL, temperature=0)
            except Exception as e:
                print(f"⚠ Gemini failed: {e}")'''
        
        print(" No LLM API key found — retrieval-only mode.")
        return None

    def ingest_data_dir(self):
        """Ingest all documents from data directory"""
        docs = load_documents()
        if docs:
            self.vdb.add_documents(docs)
            print(f" Ingested {len(docs)} documents")
        else:
            print(" No .txt files found in /data")

    def query(self, question: str, n_results: int = 6) -> Dict[str, Any]:
        """Query the RAG system"""
        # Search for relevant documents
        res = self.vdb.search(question, n_results=n_results)
        docs, metas, scores = res["documents"], res["metadatas"], res["scores"]

        if not docs:
            return {
                "answer": " No relevant context found.",
                "context": "",
                "sources": []
            }

        # Build context from top results with higher relevance threshold
        relevant_docs = []
        relevant_metas = []
        relevant_scores = []
        
        for i, score in enumerate(scores):
            if score > 0.5:  # Higher threshold for more relevant results
                relevant_docs.append(docs[i])
                relevant_metas.append(metas[i])
                relevant_scores.append(score)
        
        if not relevant_docs:
            return {
                "answer": " No sufficiently relevant context found.",
                "context": "",
                "sources": []
            }

        # Take only top 2 most relevant chunks to keep answer focused
        top_k = min(len(relevant_docs), 2)
        context_parts = []
        for i in range(top_k):
            context_parts.append(
                f"Source: {relevant_metas[i]['filename']}\n"
                f"Relevance: {relevant_scores[i]:.3f}\n"
                f"{relevant_docs[i]}"
            )
        
        context = "\n\n---\n\n".join(context_parts)[:1500]  # Reduced context size

        if not self.llm:
            return {
                "answer": " No LLM configured.",
                "context": context,
                "sources": relevant_metas[:top_k]
            }

        # Generate answer with fallback handling
        chain = self.prompt_template | self.llm | StrOutputParser()

        try:
            answer = chain.invoke({"context": context, "question": question})
            return {
                "answer": answer.strip(),
                "context": context,
                "sources": relevant_metas[:top_k]
            }
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient_quota" in error_msg or "quota" in error_msg:
                msg = " OpenAI quota exceeded. Please add GROQ_API_KEY or GOOGLE_API_KEY to .env"
            elif "rate_limit" in error_msg:
                msg = " Rate limit exceeded. Please try again in a moment."
            else:
                msg = f" LLM generation failed: {e}"
            
            return {
                "answer": msg,
                "context": context,
                "sources": relevant_metas[:top_k]
            }


def main():
    print("=== RAG Console App ===")
    app = RAGApp()
    app.ingest_data_dir()
    print("\n Ready! Ask a question (type 'exit' to quit):")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n Exiting.")
            break
        
        if not q or q.lower() in ("exit", "quit"):
            break

        result = app.query(q)
        
        print("\n--- Answer ---")
        print(result["answer"])
        
        print("\n--- Retrieved Context ---")
        print(result["context"][:1500])
        
        print("\n--- Sources ---")
        for s in result["sources"]:
            print(f"  - {s.get('filename')}")


if __name__ == "__main__":
    main()