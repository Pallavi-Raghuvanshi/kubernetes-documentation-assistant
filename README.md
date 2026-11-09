# RAG Frontier

**Engineering and evaluating retrieval-augmented generation systems.**

RAG Frontier is an experimental, end-to-end RAG project for studying how ingestion, chunking, tokenization, embeddings, retrieval, reranking, and generation affect answer quality and system performance. Kubernetes documentation is the initial corpus, not the project boundary.

## Contents

- [Pipeline](#pipeline)
- [Implemented Components](#implemented-components)
- [Retrieval Evaluation](#retrieval-evaluation)
  - [Historical Retrieval Quality](#historical-retrieval-quality)
  - [Historical Retrieval Latency](#historical-retrieval-latency)
- [Domain-Specific Tokenizer and Embedding Experiments](#domain-specific-tokenizer-and-embedding-experiments)
  - [Vocabulary Analysis](#vocabulary-analysis)
  - [Vocabulary Expansion History](#vocabulary-expansion-history)
  - [Retrieval Impact of Vocabulary Expansion](#retrieval-impact-of-vocabulary-expansion)
  - [Runtime and Memory Impact](#runtime-and-memory-impact)
- [Chunking Experiments and Data Consistency](#chunking-experiments-and-data-consistency)
- [Generation and Evaluation Measurements](#generation-and-evaluation-measurements)
  - [Recorded Generation Observations](#recorded-generation-observations)
  - [Evaluation Coverage](#evaluation-coverage)
- [Current Recorded Configuration](#current-recorded-configuration)
- [Repository Areas](#repository-areas)
- [Setup](#setup)
- [References](#references)

## Pipeline

```text
Documentation sources
        │
        ▼
Fetch / crawl → HTML extraction → Cleaning → Chunking
                                           │
                    ┌──────────────────────┴───────────────────┐
                    ▼                      ▼                  ▼
                 BM25 index          Dense embeddings      Metadata
                    └──────────────────────┬───────────────────┘
                                           ▼
                               Sparse / Dense / Hybrid
                                           │
                                  Optional reranker
                                           │
                                           ▼
                                 Context construction
                                           │
                                           ▼
                                    LLM generation
                                           │
                                           ▼
                                Grounded answer + sources
                                           │
                                           ▼
                               Retrieval / Generation evaluation
```

## Implemented components

| Area | Implementation |
|---|---|
| Ingestion | Documentation fetching, URL crawling, source URL storage, HTML text extraction and cleaning |
| Chunking | Configurable token-based chunking, overlap, validation and token inspection |
| Sparse retrieval | BM25 retrieval with a persisted local index |
| Dense retrieval | MiniLM sentence embeddings and persistent ChromaDB indexing |
| Hybrid retrieval | Sparse and dense result fusion using Reciprocal Rank Fusion (RRF) |
| Reranking | Optional cross-encoder reranker |
| Tokenizer adaptation | Corpus keyword analysis, vocabulary expansion, noise filtering and new-token embedding initialization |
| Evaluation | Retrieval quality, latency, memory, generation metrics and per-query HTML/JSONL reports |
| Generation | Local Qwen 2.5 1.5B Instruct inference experiments; streaming measurements (TTFT/TPOT) |

## Retrieval evaluation

The initial evaluation used 30 queries with chunk-level relevance labels. The query set was later expanded to **150 queries across 5 categories (30 each)**. Relevance was changed to reference paragraph text rather than fixed chunk IDs, allowing the same queries to be reused across chunking configurations.

The following six-report benchmark is the **historical 30-query baseline**, not the later 150-query evaluation.

### Historical retrieval quality

| Retrieval configuration | Recall@10 | Precision@10 | MRR | Hit Rate@10 |
|---|---:|---:|---:|---:|
| BM25 — no reranking | **0.4194** | **0.1000** | **0.4722** | **0.6000** |
| BM25 — reranking | 0.3417 | 0.0800 | 0.4431 | 0.6000 |
| Dense — no reranking | 0.2417 | 0.0600 | 0.3006 | 0.4333 |
| Dense — reranking | 0.3083 | 0.0733 | 0.4409 | 0.5333 |
| Hybrid — no reranking | 0.3694 | 0.0867 | 0.4072 | 0.5667 |
| Hybrid — reranking | 0.3250 | 0.0767 | 0.4336 | 0.5667 |

### Historical retrieval latency

| Retrieval configuration | Mean (s) | p50 (s) | p95 (s) |
|---|---:|---:|---:|
| BM25 — no reranking | 0.106 | 0.109 | 0.128 |
| Dense — no reranking | 0.015 | 0.010 | 0.017 |
| Hybrid — no reranking | 0.121 | 0.120 | 0.152 |
| BM25 — reranking | 4.786 | 4.410 | 5.232 |
| Dense — reranking | 9.392 | 9.526 | 10.727 |
| Hybrid — reranking | 9.343 | 9.628 | 10.636 |

BM25 was the strongest quality baseline in this historical benchmark. Dense retrieval was fastest but had lower quality. Hybrid retrieval did not automatically outperform BM25, and reranking added substantial CPU latency.

## Domain-specific tokenizer and embedding experiments

The original MiniLM tokenizer vocabulary contained **30,522 tokens**. Corpus keyword frequency and subword fragmentation were analyzed to identify domain terms that were poorly represented by the original tokenizer. Candidate lists were filtered to remove generated identifiers, hashes, runtime-specific names and other noisy strings.

### Vocabulary analysis

| Analysis | Count |
|---|---:|
| Unique candidate keywords analyzed | 5,864 |
| Keywords with frequency > 50 | 296 |
| Keywords with frequency > 80 | 183 |
| Keywords with frequency > 100 | 143 |
| Keywords with > 40 subwords | 19 |
| Keywords with > 30 subwords | 33 |
| Keywords with > 20 subwords | 106 |
| Keywords with frequency > 30 (later analysis) | 512 |
| Keywords with frequency > 20 (later analysis) | 765 |
| Keywords with frequency > 10 (later analysis) | 1,647 |
| Keywords with > 15 subwords (later analysis) | 195 |
| Keywords with > 10 subwords (later analysis) | 429 |
| Keywords with > 3 subwords (later analysis) | 3,182 |

### Vocabulary expansion history

| Experiment | Added candidates | Removed as unwanted | Retained added tokens | Final vocabulary |
|---|---:|---:|---:|---:|
| Initial expansion | 200 | 6 failed tokens identified initially | 200 before later filtering | 30,722 |
| Filtered expansion | 200 | 25 | 175 | 30,697 |
| Later expansion | 993 candidates selected | 153 | 840 | 31,362 |

New-token embeddings were initialized using the mean of the pretrained embeddings of each token's original subword components. The embedding dimension remained **384**. The expanded tokenizer and model were saved and used to regenerate document embeddings.

### Retrieval impact of vocabulary expansion

Results below are from the recorded 30-query experiments. The dense row distinguishes the earlier 196-token experiment from the later 840-token run.

| Retrieval | Original Recall@k | Expanded Recall@k | Original Precision@k | Expanded Precision@k | Original MRR | Expanded MRR | Original Hit Rate@k | Expanded Hit Rate@k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense — earlier expansion | 0.2417 | 0.4639 | 0.0300 | 0.0533 | 0.3006 | 0.3295 | 0.4333 | 0.6333 |
| Dense — later 840-token run | 0.2417 | 0.4500 | 0.0300 | 0.0533 | 0.3006 | 0.3037 | 0.4333 | 0.6000 |
| Hybrid — earlier expansion | 0.4694 | 0.5556 | 0.0550 | 0.0650 | 0.4128 | 0.4099 | 0.6333 | 0.7000 |
| Hybrid — later 840-token run | 0.4694 | 0.5667 | 0.0550 | 0.0667 | 0.4128 | 0.4099 | 0.6333 | 0.7000 |
| Sparse — vocabulary experiment | 0.4972 | 0.4972 | 0.0583 | 0.0583 | 0.4743 | 0.4743 | 0.6333 | 0.6333 |

### Runtime and memory impact

| Retrieval | Original latency (s) | Expanded latency (s) | Original peak memory (MB) | Expanded peak memory (MB) |
|---|---:|---:|---:|---:|
| Dense — earlier expansion | 0.0145 | 0.0884 | 546.375 | 544.9107 |
| Dense — later 840-token run | 0.0145 | 0.1085 | 546.375 | 550.4783 |
| Hybrid — earlier expansion | 0.1205 | 0.1799 | 695.020 | 695.7243 |
| Sparse — earlier expansion | 0.1056 | 0.1107 | 503.521 | 503.5913 |

Vocabulary expansion improved dense and hybrid retrieval in the recorded experiments, while sparse retrieval was unchanged because BM25 does not use the embedding tokenizer. Dense retrieval latency increased in these runs; results should be interpreted as experiment-specific rather than proof that larger vocabularies always improve retrieval.

## Chunking experiments and data consistency

For 512-token chunks, the following configurations were recorded:

| Chunk size | Overlap | Overlap tokens | Chunk count | Recall@20 | Precision@20 | MRR | Hit Rate@20 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 25% | 128 | 27,193 | 0.4500 | 0.0533 | 0.3037 | 0.6000 |
| 512 | 5% | 26 | 22,556 | 0.3000 | 0.0367 | 0.1370 | 0.4000 |
| 512 | 10% | 51 | 23,477 | 0.3000 | 0.0367 | 0.1370 | 0.4000 |
| 512 | 20% | 102 | 25,827 | Not recorded | Not recorded | Not recorded | Not recorded |


A consistency issue occurred when chunks were regenerated but embeddings from the previous chunking configuration were reused: the new chunk count was **22,556**, while the old embedding set contained **27,193** rows.

Whenever chunking changes, rebuild in this order:

1. Generate chunks.
2. Generate embeddings from those exact chunks.
3. Rebuild the ChromaDB index from those embeddings.
4. Verify that chunk, embedding and index record counts match.
5. Run retrieval evaluation.

Embedding-generation and indexing scripts were updated to select the configured embedding file, and a ChromaDB verification script was added.

## Generation and evaluation measurements

The local Qwen 2.5 1.5B Instruct model was explored for answer generation. Early CPU inference was slow and produced incomplete answers.

### Recorded generation observations

| Measurement | Recorded result |
|---|---:|
| Early single-query generation latency | 252.499 s |
| Later observed generation time | Approximately 1 minute per query |
| Early faithfulness score | 0.0000 |
| Early answer completeness score | 0.0000 |
| Early hallucination rate | 0.9333 |

The log notes that weak generation results may reflect the local model, the small judge model, or evaluation interpretation. They require further validation.

### Evaluation coverage

| Evaluation area | Metrics / outputs |
|---|---|
| Retrieval quality | Recall@k, Precision@k, MRR, Hit Rate@k |
| Runtime | Retrieval latency, generation latency, TTFT, TPOT, peak memory |
| Generation quality | Faithfulness, hallucination rate, answer completeness |
| Query inspection | Retrieved chunks, reference-context token comparison, per-query metrics |
| Artifacts | JSONL query outputs and HTML reports |

The latest logged run used **1 query** and reported retrieval-only measurements: retrieval latency **0.1038 s**, total measured time **0.1064 s**, peak memory approximately **564 MB**, reference text coverage **1.0**, and retrieved text relevance **0.0297**. This is a single-query diagnostic, not a full benchmark or generated-answer evaluation.

## Current recorded configuration

| Item | Recorded value |
|---|---|
| Initial corpus | Kubernetes documentation |
| Dense model | `sentence-transformers/all-MiniLM-L6-v2` and expanded-vocabulary variant |
| Original tokenizer vocabulary | 30,522 tokens |
| Latest expanded tokenizer vocabulary | 31,362 tokens |
| Added tokens retained | 840 |
| Embedding dimension | 384 |
| Latest logged chunk configuration | 512 tokens, 5% overlap (26 tokens) |
| Latest matching chunk and embedding count | 22,556 |
| Vector store | Persistent ChromaDB |
| Hybrid fusion | Reciprocal Rank Fusion |
| Optional reranker | Cross-encoder |
| Evaluation query set | 150 queries, 5 categories |
| Generation model explored | Qwen 2.5 1.5B Instruct |

## Repository areas

```text
configs/       Corpus, retrieval, model, and experiment configuration
data/          Raw and processed corpus, chunks, queries, embeddings, indexes
docs/          Experiment and query-generation notes
experiments/   Query outputs and experiment records
models/        Downloaded and adapted model/tokenizer artifacts
scripts/       Chunking, indexing, verification, and report utilities
src/
  ingestion/   Fetching, crawling, HTML extraction, chunking, validation
  tokenization/Tokenizer utilities
  embeddings/  Embedding generation and vocabulary experiments
  retrieval/   BM25, dense, and hybrid retrieval
  reranker/    Cross-encoder reranking
  evaluation/  Retrieval and generation metrics
  generation/  Local LLM inference and generation workflow
  rag_pipeline.py
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure corpus and experiment settings before running the pipeline. Use the module and script names present in the current repository, as some original numbered scripts were later reorganized.

## References

- [Sentence Transformers](https://www.sbert.net/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [ChromaDB documentation](https://docs.trychroma.com/)
- [Domain-specific tokenizer vocabulary expansion](https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41)
