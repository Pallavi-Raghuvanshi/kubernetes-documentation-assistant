# rag_pipeline.py
import time
import yaml

from src.generation._6llm_inference import generate_answer
from pathlib import Path
from src.retrieval.__4bm25_index import search as bm25_search, load_index
from src.retrieval.dense_retrieval import search as dense_search

def run_pipeline(
    question: str,
    config: dict,
    index=None,
    memory_monitor_factory=None,
    run_generation: bool = True,
) -> dict:

    retrieval_memory_monitor = (
        memory_monitor_factory()
        if memory_monitor_factory is not None
        else None
    )

    if retrieval_memory_monitor:
        retrieval_memory_monitor.start()

    retrieval_start = time.perf_counter()
    retriever = config["retrieval"].get("default_retriever", "bm25").lower()
    top_k = config["retrieval"]["top_k"]

    if retriever == "bm25":
        chunks, bm25 = index
        results = bm25_search(
            query=question,
            chunks=chunks,
            bm25=bm25,
            top_k=top_k,
        )

    elif retriever == "dense":
        chunks, embeddings, model = index
        results = dense_search(
            query=question,
            chunks=chunks,
            embeddings=embeddings,
            model=model,
            top_k=top_k,
        )

    else:
        raise ValueError(f"Unsupported retriever: {retriever}")
    retrieval_latency = time.perf_counter() - retrieval_start

    retrieval_peak_memory_mb = (
        retrieval_memory_monitor.stop()
        if retrieval_memory_monitor
        else None
    )

    context_parts = []

    for result in results:
        context_parts.append(
            f"[Source {result['rank']}] {result['source']}\n"
            f"{result['text']}"
        )

    context = "\n\n".join(context_parts)

    if not run_generation:
        return {
            "results": results,
            "context": context,
            "generation_result": None,
            "retrieval_latency_seconds": retrieval_latency,
            "retrieval_peak_memory_mb": retrieval_peak_memory_mb,
            "generation_peak_memory_mb": None,
        }

    generation_memory_monitor = (
        memory_monitor_factory()
        if memory_monitor_factory is not None
        else None
    )

    if generation_memory_monitor:
        generation_memory_monitor.start()

    generation_result = generate_answer(
        question=question,
        context=context,
        max_new_tokens=config["generation"]["max_new_tokens"],
    )

    generation_peak_memory_mb = (
        generation_memory_monitor.stop()
        if generation_memory_monitor
        else None
    )

    return {
        "results": results,
        "context": context,
        "generation_result": generation_result,
        "retrieval_latency_seconds": retrieval_latency,
        "retrieval_peak_memory_mb": retrieval_peak_memory_mb,
        "generation_peak_memory_mb": generation_peak_memory_mb,
    }



def main():
    root = Path(__file__).resolve().parents[1]

    with (root / "queries.yaml").open(encoding="utf-8") as file:
        queries = yaml.safe_load(file)["queries"]

    with (root / "configs.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # Limit queries using evaluation.num_queries.
    num_queries = int(config.get("evaluation", {}).get("num_queries", 0))

    retriever = config["retrieval"].get("default_retriever", "bm25").lower()

    if retriever == "bm25":
        chunks, bm25 = load_index()
        index = (chunks, bm25)

    elif retriever == "dense":
        from sentence_transformers import SentenceTransformer
        from src.embeddings.embeddings import MODEL_NAME
        from src.retrieval.dense_retrieval import load_dense_index

        chunks, embeddings = load_dense_index()
        model = SentenceTransformer(MODEL_NAME)
        index = (chunks, embeddings, model)

    else:
        raise ValueError(f"Unsupported retriever: {retriever}")

    queries = queries[:num_queries]

    pipeline_result = run_pipeline(
        question=queries[0]["query"],
        config=config,
        index=index,
    )

    results = pipeline_result["results"]
    generation_result = pipeline_result["generation_result"]

    print("\n--- Answer ---")
    print(generation_result["answer"])

    print("\n--- Retrieved Sources ---")
    seen_sources = set()

    for result in results:
        if result["source"] not in seen_sources:
            seen_sources.add(result["source"])
            print(f"- {result['source']}")

    print("\n--- Latency ---")
    print(f"Retrieval: {pipeline_result['retrieval_latency_seconds']:.4f} seconds")
    print(f"Generation: {generation_result['generation_latency_seconds']:.3f} seconds")
