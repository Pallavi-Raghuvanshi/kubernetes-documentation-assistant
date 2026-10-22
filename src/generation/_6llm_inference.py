# src/generation/_6llm_inference.py
import time
import torch
import threading
from transformers import TextIteratorStreamer

from src.models._5load_models import load_model

# Load lazily: retrieval-only evaluation does not need Qwen.
tokenizer = None
model = None
_model_lock = threading.Lock()


def get_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        with _model_lock:
            if tokenizer is None or model is None:
                tokenizer, model = load_model()

    return tokenizer, model


class TimingTextIteratorStreamer(TextIteratorStreamer):
    """Track generated token IDs and the first generated token time."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_token_time = None
        self.generated_token_count = 0
        self._first_put_is_prompt = True

    def put(self, value):
        # The first call contains prompt tokens, not generated tokens.
        if self._first_put_is_prompt:
            self._first_put_is_prompt = False
        else:
            self.generated_token_count += value.shape[-1]

            if self.first_token_time is None:
                self.first_token_time = time.perf_counter()

        super().put(value)

def generate_answer(
    question: str,
    context: str = "",
    max_new_tokens: int = 160,
) -> dict:

    tokenizer, model = get_model()
    user_prompt = (
        "Answer using only the supplied context. "
        "If the context is insufficient, say so. "
        "Do not invent facts.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}"
    )

    messages = [
        {
            "role": "system",
            "content": "You are a concise and accurate assistant.",
        },
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")


    input_length = inputs["input_ids"].shape[1]

    streamer = TimingTextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    errors = []

    def run_generation():
        try:
            with torch.inference_mode():
                model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer,
                )
        except Exception as exc:
            errors.append(exc)
            streamer.end()

    start = time.perf_counter()

    generation_thread = threading.Thread(
        target=run_generation,
        daemon=True,
    )
    generation_thread.start()

    text_chunks = list(streamer)
    generation_thread.join()

    if errors:
        raise RuntimeError("Model generation failed.") from errors[0]

    generation_time = time.perf_counter() - start
    answer = "".join(text_chunks).strip()
    generated_tokens = streamer.generated_token_count

    ttft_seconds = (
        streamer.first_token_time - start
        if streamer.first_token_time is not None
        else None
    )

    tpot_ms = (
        max(0.0, generation_time - ttft_seconds)
        / (generated_tokens - 1)
        * 1000
        if ttft_seconds is not None and generated_tokens > 1
        else None
    )

    return {
        "answer": answer,
        "generation_latency_seconds": round(generation_time, 4),
        "generated_tokens": generated_tokens,
        "input_tokens_count": input_length,
        "ttft_seconds": (
            round(ttft_seconds, 4)
            if ttft_seconds is not None
            else None
        ),
        "tpot_ms": (
            round(tpot_ms, 3)
            if tpot_ms is not None
            else None
        ),
    }


if __name__ == "__main__":
    result = generate_answer(
        question="What is RAG? Explain in three short bullet points.",
        max_new_tokens=100,
    )
    print("\n--- Answer ---")
    print(result["answer"]+"\n")
    print(f"         Latency     : {result['generation_latency_seconds']} seconds")
    print(f"Generated Token count: {result['generated_tokens']}")
    print(f"            TTFT     : {result['ttft_seconds']} seconds")
    print(f"            TPOT     : {result['tpot_ms']} ms/token")
