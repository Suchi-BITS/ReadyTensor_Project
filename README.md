# RAG Console Application

A modular Retrieval-Augmented Generation (RAG) system using LangChain, ChromaDB, HuggingFace embeddings, and CrossEncoder reranking. This project supports document ingestion, vector search, reranking, and LLM response generation through OpenAI and Groq.

---

## Features

- RAG pipeline with vector search and semantic reranking
- Automatic ingestion of .txt documents from the data folder
- Persistent ChromaDB vector database
- High-quality HuggingFace MiniLM embeddings
- Token-based and paragraph-based text chunking
- LLM fallback chain: OpenAI → Groq → Retrieval-only
- Clean architecture with separate modules for vector DB and app logic
- Environment-driven configuration through .env

---

## Project Structure

```
project/
│
├── data/                      # Input documents (.txt files)
│       └── *.txt
│
├── chroma_store/              # Auto-persisted ChromaDB files
│
├── src/
│       ├── app.py             # Main RAG application
│       └── vectordb.py        # Vector DB wrapper (Chroma + HF + reranker)
│
├── .env                       # Configuration keys and model selection
└── README.md
```

---

## Architecture Overview

### 1. Document Ingestion
- Loads all .txt files from the data directory.
- Splits documents into meaningful chunks:
  - Paragraph-based chunking
  - Token-based chunking using RecursiveCharacterTextSplitter

### 2. Embedding Layer
- Generates normalized embeddings using HuggingFace MiniLM.
- Implemented via langchain-huggingface.

### 3. Vector Store
- Stores embeddings using Chroma from langchain-chroma.
- Data persists automatically inside the chroma_store directory.

### 4. Reranking Layer
- Uses a CrossEncoder reranker (ms-marco-MiniLM-L-6-v2) to reorder retrieved documents and improve answer accuracy.

### 5. LLM Response Layer
- Priority order:
  1. OpenAI (gpt-4o-mini)
  2. Groq (llama-3.1-8b-instant)
  3. Retrieval-only fallback if no API keys are available

### 6. Console Interface
- Simple REPL-style question-and-answer interface.
- Displays answer, retrieved context, and document sources.

---

## Required .env Configuration

Create a .env file in the project root:

```
# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4o-mini

# Groq (optional fallback)
GROQ_API_KEY=your_groq_key
GROQ_MODEL=llama-3.1-8b-instant

# Embedding model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Chroma collection name
CHROMA_COLLECTION_NAME=rag_documents
```

---

## Installation

### 1. Clone the repository
```
git clone <your_repo_url>
cd <project_folder>
```

### 2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Upgrade pip
```
python -m pip install --upgrade pip
```

### 4. Install dependencies

Create requirements.txt containing:

```
langchain
langchain-core
langchain-community
langchain-chroma
langchain-huggingface
langchain-text-splitters

chromadb
sentence-transformers
numpy
tiktoken

openai
groq

python-dotenv
requests
pydantic
typing-extensions
```

Install them:

```
pip install -r requirements.txt
```

---

## Running the Application

```
python src/app.py
```

---

## How It Works

1. Loads all .txt documents from the data folder
2. Splits them into optimized chunks
3. Generates embeddings using HuggingFace
4. Stores vectors in ChromaDB
5. Retrieves relevant chunks for a query
6. Reranks them using CrossEncoder
7. Constructs final context
8. Generates an answer using available LLMs

---

## Resetting the Vector Store

To clear Chroma:

```
from vectordb import VectorDB
v = VectorDB("rag_documents", "sentence-transformers/all-MiniLM-L6-v2")
v.clear_collection()
```

Or delete the chroma_store folder manually.

---

## Notes

- Only .txt files are supported for ingestion.
- Restart the application after adding new documents.
- All components run on CPU by default.
- You can switch to larger embedding models if required.
