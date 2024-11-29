import os
import time

from dotenv import load_dotenv
from optimum.pipelines import pipeline

# Load the .env file
load_dotenv()


chat_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-1B-Instruct",
    accelerator="ort",
    token=os.getenv("HUGGINGFACE_API_TOKEN"),
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"},
]

print("start")
for i in range(10):
    start = time.time()
    # Generate response
    generated = chat_pipeline(
        messages, max_length=1024, num_return_sequences=1, truncation=True
    )  # type: ignore

    end = time.time()

    print(generated)
    print(end - start)
