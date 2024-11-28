from dotenv import load_dotenv
import time

from llama_cpp import Llama

from nilai.crypto import generate_key_pair
from nilai.model import Model


load_dotenv()


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()
        self.chat_pipeline = Llama.from_pretrained(
            repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
            filename="Llama-3.2-1B-Instruct-Q5_K_S.gguf",
        )
        self.models = [
            Model(
                id="meta-llama/Llama-3.2-1B-Instruct",
                name="Llama-3.2-1B-Instruct",
                version="1.0",
                description="Llama is a large language model trained on supervised and unsupervised data.",
                author="Meta-Llama",
                license="Apache 2.0",
                source="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct",
                supported_features=["text-completion", "chat"],
            )
        ]
        self._uptime = time.time()

    @property
    def uptime(self):
        elapsed_time = time.time() - self._uptime
        days, remainder = divmod(elapsed_time, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if days > 0:
            parts.append(f"{int(days)} days")
        if hours > 0:
            parts.append(f"{int(hours)} hours")
        if minutes > 0:
            parts.append(f"{int(minutes)} minutes")
        if seconds > 0:
            parts.append(f"{int(seconds)} seconds")

        return ", ".join(parts)


state = AppState()
