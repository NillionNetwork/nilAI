from llama_cpp import Llama
import time
llm = Llama.from_pretrained(
	repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
	filename="Llama-3.2-1B-Instruct-Q5_K_S.gguf",
)


messages = [    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What is your name?"
    }]

print("start")
for i in range(10):
    start = time.time()
    # Generate response
    generated = llm.create_chat_completion(messages)

    end = time.time()

    print(generated)
    print(end - start)