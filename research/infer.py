import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
logging.disable(logging.WARNING)

MODEL_PATH = "/workspace/models/llama-3.1-8b-hf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

def infer(prompt: str, system_prompt: str = None, max_new_tokens: int = 100) -> str:
    if system_prompt:
        full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        full_prompt = prompt

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    print("Model loaded. Running test...\n")
    response = infer("Hi, how are you?")
    print(f"Prompt: Hi, how are you?")
    print(f"Response: {response}")
