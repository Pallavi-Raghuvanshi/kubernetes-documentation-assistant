# RAG Frontier: CPU-Optimized Kubernetes RAG Assistant

### **Project objective:**

Build a Kubernetes AI assistant grounded in documentation and identify the retrieval configuration that delivers the best measured relevance within practical CPU latency and memory constraints.

### Key Features

* Kubernetes documentation ingestion and chunking.
* Sparse retrieval using BM25.
* Dense semantic retrieval using MiniLM embeddings.
* Hybrid retrieval using Reciprocal Rank Fusion (RRF).
* Optional cross-encoder reranking.
* Domain-specific tokenizer vocabulary expansion.
* Evaluation using query-level relevance judgments.
* Retrieval latency and memory profiling.
* Planned experiments with alternative embedding models, vector indexes, and numerical precision.

## 2. Experimental Progress

| Stage | Experiment                         | Purpose                                                   |
| ----- | ---------------------------------- | --------------------------------------------------------- |
| 1     | BM25 sparse retrieval              | Establish a lexical retrieval baseline                    |
| 2     | MiniLM dense retrieval             | Evaluate semantic retrieval                               |
| 3     | Hybrid retrieval using RRF         | Combine lexical and semantic signals                      |
| 4     | Cross-encoder reranking            | Test whether reranking improves result ordering           |
| 5     | MiniLM vocabulary expansion        | Investigate domain-specific tokenization                  |
| 6     | New-token embedding initialization | Initialize added tokens using original subword embeddings |
| 7     | Fine-tuning expanded embeddings    | Planned investigation of retrieval-quality improvements   |

## 3. Dataset and Model Statistics

| Metric                                            | Observed value |
| ------------------------------------------------- | -------------- |
| Unique tokens extracted from corpus               | 20,775         |
| Corpus chunks represented in the embedding matrix | 27,193         |
| Embedding dimensions                              | 384            |
| Embedding data type                               | FP32           |
| Embedding matrix size                             | 27,193 × 384  |
| Original MiniLM vocabulary                        | 30,522 tokens  |
| Expanded vocabulary                               | 30,722 tokens  |
| Additional tokens                                 | 200            |
| New tokens initialized using subword means        | 194            |
| New tokens not initialized using subword means    | 6              |
| Evaluation queries in current benchmark           | 30             |

The six tokens that could not be initialized using subword means were identified as large Base64-like strings associated with Mermaid diagram content. Their inclusion and treatment require further investigation.

**Embedding storage:** The FP32 matrix contains 10,442,112 values, occupying approximately 39.8 MiB of raw numerical data, excluding metadata and other runtime allocations.

## 4. Retrieval Benchmark Results

The following results were calculated from the six supplied JSONL evaluation reports, each containing 30 queries.

### Retrieval Quality

| Retrieval configuration        | Recall@10        | Precision@10     | MRR              | Hit Rate@10      |
| ------------------------------ | ---------------- | ---------------- | ---------------- | ---------------- |
| **BM25 — No reranking** | **0.4194** | **0.1000** | **0.4722** | **0.6000** |
| BM25 — Reranking              | 0.3417           | 0.0800           | 0.4431           | 0.6000           |
| Dense — No reranking          | 0.2417           | 0.0600           | 0.3006           | 0.4333           |
| Dense — Reranking             | 0.3083           | 0.0733           | 0.4409           | 0.5333           |
| Hybrid — No reranking         | 0.3694           | 0.0867           | 0.4072           | 0.5667           |
| Hybrid — Reranking            | 0.3250           | 0.0767           | 0.4336           | 0.5667           |

**Metric definitions**

* **Recall@10:** Fraction of relevant chunks retrieved within the top 10.
* **Precision@10:** Fraction of the top 10 results that are relevant.
* **MRR:** Mean reciprocal rank of the relevant chunk retrieved first.
* **Hit Rate@10:** Fraction of queries with at least one relevant result in the top 10.

*These results describe the supplied evaluation reports; they do not establish the cause of every performance difference. The reranking experiments must also be interpreted in light of candidate selection and reranker configuration.*

### CPU Performance

| Retrieval configuration | Mean latency (s) | p50 latency (s) | p95 latency (s) |
| ----------------------- | ---------------- | --------------- | --------------- |
| BM25 — No reranking    | 0.106            | 0.109           | 0.128           |
| Dense — No reranking   | 0.015            | 0.010           | 0.017           |
| Hybrid — No reranking  | 0.121            | 0.120           | 0.152           |
| BM25 — Reranking       | 4.786            | 4.410           | 5.232           |
| Dense — Reranking      | 9.392            | 9.526           | 10.727          |
| Hybrid — Reranking     | 9.343            | 9.628           | 10.636          |

*Latency statistics are calculated from the supplied reports. These are observed benchmark results, not guaranteed performance on other hardware. Reranking adds substantial latency in the current setup.*

### Key Findings

1. **BM25 is the strongest current baseline.** It achieved Recall@10 of 0.4194 and MRR of 0.4722 without reranking.
2. **Dense retrieval is faster in the current measurements but has lower retrieval quality.** Its Recall@10 is 0.2417.
3. **Hybrid retrieval did not outperform BM25.** Its Recall@10 of 0.3694 suggests that combining retrieval signals does not automatically improve relevance.
4. **Reranking effects vary by retrieval strategy.** Dense retrieval's MRR improved from 0.3006 to 0.4409, while BM25's MRR decreased from 0.4722 to 0.4431.
5. **Reranking has a substantial latency cost.** Further investigation is needed to determine whether its quality improvements justify the additional computation.
6. **The tokenizer experiment remains an open research question.** Vocabulary expansion and embedding initialization have been implemented, but their retrieval-quality impact must be established through a controlled evaluation.

## 5. Problems Encountered and Engineering Decisions

| Problem                                                                    | Investigation or solution                                                                                                                    |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense retrieval underperformed BM25                                        | Compared sparse, dense, and hybrid retrieval using the same 30 evaluation queries                                                            |
| Hybrid retrieval did not consistently improve results                      | Evaluated RRF rather than assuming combined retrieval would be superior                                                                      |
| Reranking increased latency substantially                                  | Benchmarked each retrieval strategy with and without reranking                                                                               |
| Kubernetes terminology may be poorly represented by the original tokenizer | Analyzed keyword frequency and subword fragmentation to select candidate tokens                                                              |
| Some candidate tokens appeared to be noisy strings                         | Identified Base64-like content, hashes, UUIDs, and IDs for further filtering                                                                 |
| Added token embeddings needed initialization                               | Initialized 194 of 200 added tokens using mean original subword embeddings                                                                   |
| Expanded tokenizer was not initially loaded correctly after saving         | Saved the expanded tokenizer at the SentenceTransformer root and Transformer subdirectory, then verified vocabulary and embedding dimensions |
| Dense retrieval expected a model object but could receive a string         | Corrected the query-embedding call to use `<span>SentenceTransformer.encode()</span>`                                                      |
| Different models require different document embeddings                     | Designed the evaluation to generate embeddings separately for each model                                                                     |
| Evaluation results could be contaminated by training                       | Reserved the existing 30 evaluation queries and planned separate training query–positive-chunk pairs                                        |

### Tokenizer Vocabulary Selection

The keyword analysis produced the following observations:

| Selection criterion                                      | Number of keywords |
| -------------------------------------------------------- | ------------------ |
| Total unique keywords analyzed                           | 5,864              |
| Occurrences > 50                                         | 296                |
| Occurrences > 80                                         | 183                |
| Occurrences > 100                                        | 143                |
| Subword count > 40                                       | 19                 |
| Subword count > 30                                       | 33                 |
| Subword count > 20                                       | 106                |
| Overlap between occurrences > 100 and subword count > 20 | 0                  |

The analysis identified frequently occurring terms and highly fragmented terms as distinct candidate groups. The expanded vocabulary was constructed from selected candidates in these groups.

**Next step:** Compare original MiniLM, initialized MiniLM, and trained MiniLM using identical chunks, evaluation queries, relevance judgments, and retrieval settings.

## 6. Evaluation Methodology

The benchmark uses 30 evaluation queries with manually specified relevant chunk IDs.

For controlled comparisons:

* Keep the corpus and chunk boundaries fixed.
* Keep evaluation queries and relevance judgments unchanged.
* Generate document embeddings separately for every embedding model.
* Encode each query using the corresponding model.
* Use consistent retrieval cutoffs and metrics.
* Disable reranking when isolating the effect of the embedding model.
* Report aggregate and per-query results.
* Record latency, memory consumption, and index size alongside quality.

Training queries must remain separate from evaluation queries.

## 7. Project Structure

```
rag_frontier/
├── configs/
│   └── corpus.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── embeddings/
│   └── training/
├── models/
│   ├── kubernetes_updated_tokenizer_vocab/
│   └── kubernetes_minilm_initialized/
├── indexes/
├── src/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── reranker/
│   ├── evaluation/
│   ├── generation/
│   └── rag_pipeline.py
├── experiments/
│   └── query_outputs/
├── reports/
├── tests/
├── scripts/
├── queries.yaml
├── configs.yaml
├── requirements.txt
└── README.md
```

## 8. Execution and Configuration

Create and activate the environment on Windows:

```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure the corpus in `<span>configs/corpus.yaml</span>` and retrieval settings in `<span>configs.yaml</span>`.

The current workflow covers:

1. Fetch and extract Kubernetes documentation.
2. Chunk the corpus.
3. Build the BM25 index.
4. Generate dense embeddings.
5. Run sparse, dense, or hybrid retrieval.
6. Evaluate retrieval quality.
7. Run tokenizer and embedding experiments.
8. Compare quality, latency, and memory.
9. Analyze per-query failures.

Check the current source filenames before running individual stages, as the implementation has evolved beyond the original numbered scripts.

## 9. Next Experiments

| Priority | Experiment                                          | Objective                                                     |
| -------- | --------------------------------------------------- | ------------------------------------------------------------- |
| 1        | Evaluate original versus initialized MiniLM         | Measure the effect of vocabulary expansion                    |
| 2        | Fine-tune added token embeddings                    | Determine whether training improves retrieval quality         |
| 3        | Investigate failed queries                          | Identify terminology, chunking, or ranking failure patterns   |
| 4        | Reassess reranking                                  | Improve the quality–latency trade-off                        |
| 5        | Compare BGE and other embedding models              | Find stronger semantic retrieval candidates                   |
| 6        | Compare FAISS Flat, HNSW, and IVF-Flat              | Evaluate indexing speed, search latency, and memory           |
| 7        | Test FP32, FP16, and INT8                           | Measure precision and storage trade-offs where supported      |
| 8        | Build a Pareto frontier                             | Identify configurations that satisfy CPU resource constraints |
| 9        | Integrate the selected retriever into the assistant | Complete the Kubernetes RAG application                       |

## 10. References

* [Domain-specific tokenizer vocabulary expansion](https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41)
* [Sentence Transformers](https://www.sbert.net/)
* [Hugging Face Transfo](https://huggingface.co/docs/transformers/index)

## 11. Regenerate records using git hash

To Temporarily Switch your Workspace to a Commit

```
git switch --detach <commit-hash>
```

To go back to current directory

```
git switch -
```
