import os
import time
from asyncio import Semaphore

from dotenv import load_dotenv
from llama_cpp import Llama

from nilai.crypto import generate_key_pair
from nilai.model import Model


class AppState:
    def __init__(self):
        self.private_key, self.public_key, self.verifying_key = generate_key_pair()

        model_path = os.environ["MODEL_PATH"]
        model_filename = os.environ["MODEL_FILENAME"]

        self.chat_pipeline = Llama.from_pretrained(
            repo_id=model_path,
            filename=model_filename if model_filename else None,
            n_threads=16,
            verbose=False,
        )
        self.sem = Semaphore(2)
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


load_dotenv()
state = AppState()
