
import time

import torch

from src.models._5load_models import load_model

# Load once and reuse for every request.
tokenizer, model = load_model()


def generate_answer(
    question: str,
    context: str = "",
    max_new_tokens: int = 160,
) -> dict:
    
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

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generation_time = time.perf_counter() - start
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0, input_length:]

    return {
        "answer": tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
        "generation_latency_seconds": round(generation_time, 3),
        "generated_tokens": len(new_tokens),
        "input_tokens_count": input_length,
    }


if __name__ == "__main__":
    result = generate_answer(
        question="What is RAG? Explain in three short bullet points.",
        max_new_tokens=100,
    )
    print("\n--- Answer ---")
    print(result["answer"])
    print(f"\nLatency: {result['generation_latency_seconds']} seconds")
    print(f"Tokens generated: {result['generated_tokens']}")
