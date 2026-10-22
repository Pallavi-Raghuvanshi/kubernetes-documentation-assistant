# RAG Retrieval Quality–Latency–Memory Frontier

## Introduction

This project benchmarks Retrieval-Augmented Generation (RAG) retrieval systems to analyze trade-offs between **retrieval quality, latency, and memory consumption** across different retrieval models, indexing strategies, and numerical precisions. Kubernetes documentation has been used as corpus which can be updated in ``configs/corpus.yaml``

**Goal:** is to identify optimal retrieval configurations under specific performance and hardware constraints.

## Research Questions

1. ~How do retrieval models and index types affect Recall@10 and latency?~
2. ~How does FP32, FP16, and INT8 precision impact recall and memory usage?~
3. Which configuration maximizes retrieval quality under latency and memory constraints?

## Notes

1. `_1<filename>` means code file to be run at step 1
2. `__1<filename>` means that this file has to be run only once after file run at step 1
3. `_v1<filename>` means this code file is just for validation of output of code file run at step 1

## Imp. Concepts

- **BM25 Index** maintains
  - Metadata of each chunk:  `chunk_id `, `source `, `chunk_size `, `text`
  - Term frequencies in each chunk
  - IDF (Inverse Document Frequency): Less often the word appears across docs, more unique info it carries
  - Document lengths: total tokens in each chunk

## Indexing results

* 20,775 unique tokens extracted from whole corpus

## Experimental Scope

Benchmarking RAG retrieval configurations across quality, latency, and memory.

* **Chunking: Fixed 512-token chunks with 128-token overlap**
* **Tokenization:** Tokenizer behavior • Structure-aware chunking
* **Embedding Models:** MiniLM • BGE • Embedding generation cost
* **Retrieval Strategies:** Sparse Retrieval (BM25) • Dense retrieval • Hybrid retrieval (RRF)     < - applied on chunks
* **Vector Indexes:** FAISS Flat • HNSW • IVF-Flat
* **Precision:** FP32 • FP16 • INT8 (where supported)
* **Evaluation:**
  * **Recall@10**: Proportion of relevant chunks retrieved in top 10 retrieved chunks
  * **Precision@10:** Proportion of chunks retrieved which are relevant to the query.
  * **MRR:** Mean of Reciprocal of Rank of the 1st relevant chunk for each of the query token
  * **Hit Rate@10:** Proportion of queries having atleast one relevant chunk in top 10 retrieved chunks
* **Performance:** p50/p95/p99 latency • Index size • Runtime memory
* **Analysis:** Pareto frontier • Quality–latency–memory trade-offs • Constraint-aware configuration selector

**Scope:** Focuses on retrieval benchmarking, not LLM answer generation.

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

### 4. Build BM25 Index for the corpus once

```python
python src/retrieval/_4bm25_index.py
```

### 5. BM25 retrieval

```code
python src/retrieval/5bm25_retriever.py
```

### 6. Evaluates the BM25 results using Recall@10, MRR, and Hit Rate

```code
python src/evaluation/6evaluate_retrieval.py
```

## Download LLM

```python
python src\models\_6download_models.py
```

### 7. Load LLM

```python
run python src\models\7load_models.py
```

### 8. Inference (generate text response)

```python
python src.generation._8llm_inference
```
