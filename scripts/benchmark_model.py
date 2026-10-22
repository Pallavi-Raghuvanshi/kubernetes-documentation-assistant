import time
import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(1)


from src.models._5load_models import load_model

print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Interop threads: {torch.get_num_interop_threads()}")

start = time.perf_counter()
tokenizer, model = load_model()
load_time = time.perf_counter() - start

print(f"\nModel loading time: {load_time:.2f} seconds")

# Warm-up and timed generation
prompt = "Explain Kubernetes API Priority and Fairness briefly."
inputs = tokenizer(prompt, return_tensors="pt")

with torch.inference_mode():
    model.generate(**inputs, max_new_tokens=20)

start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=100)

generation_time = time.perf_counter() - start

print(f"Generation time: {generation_time:.2f} seconds")
print(f"Generated tokens: {output.shape[1] - inputs['input_ids'].shape[1]}")
