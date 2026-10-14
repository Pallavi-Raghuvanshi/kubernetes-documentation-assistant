# RAG Retrieval Quality–Latency–Memory Frontier

## Introduction

This project benchmarks Retrieval-Augmented Generation (RAG) retrieval systems to analyze trade-offs between **retrieval quality, latency, and memory consumption** across different retrieval models, indexing strategies, and numerical precisions.

The goal is to identify optimal retrieval configurations under specific performance and hardware constraints. 

Kubernetes documentation has been used as corpus which can be updated in ``configs/corpus.yaml``

## Research Questions

1. How do retrieval models and index types affect Recall@10 and latency?
2. How does FP32, FP16, and INT8 precision impact recall and memory usage?
3. Which configuration maximizes retrieval quality under latency and memory constraints?

## Project Structure

```
rag-frontier/
├── configs/
├── data/
├── indexes/
├── src/
├── experiments/
├── reports/
├── tests/
├── scripts/
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation and Setup

### 1. Clone Repository & Create Virtual Environment

```code
git clone https://github.com/your-username/RAGfrontier.git
cd rag_frontier

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (Linux/macOS)
source .venv/bin/activate
```

### 2. Install Dependencies

```code
pip install -r requirements.txt
```

## Usage & Execution

### 1. Fetch documentation

```code
python -m src.ingestion.1fetch_docs
```

### 2. Extract text from HTML file

```code
python -m src.ingestion.2extract_text
```

### 3. Token-based chunking

```code
python -m src.ingestion.3chunk_text
```

### 4. Validate Chunks

```code
python src/ingestion/4validate_chunks.py
```

### 5. BM25 retrieval

```code
python src/retrieval/5bm25_retriever.py
```

### 6. Evaluates the BM25 results using Recall@10, MRR, and Hit Rate

```code
python src/evaluation/6evaluate_retrieval.py
```
