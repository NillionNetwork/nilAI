import os
import time

import torch
from dotenv import load_dotenv
from transformers import pipeline


# Load the .env file
load_dotenv()

# # Application State Initialization
torch.set_num_threads(32)
torch.set_num_interop_threads(32)


chat_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cpu",
    token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"},
]

start = time.time()
# Generate response
generated = chat_pipeline(
    messages, max_length=1024, num_return_sequences=1, truncation=True
)  # type: ignore

end = time.time()

print(generated)
print(end - start)
