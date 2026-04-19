import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B-Chat"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("Model and tokenizer loaded successfully.")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Introduce yourself."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

print("Encoded input text:")
print(model_inputs)

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=512
)

# Get only the generated part of the output (excluding the input tokens)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the generated tokens to get the response text
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("\nModel Response:")
print(response)