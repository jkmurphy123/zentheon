from __future__ import annotations
from typing import List, Dict
from rich import print
from utils import load_config

cfg = load_config()

class LLMClient:
    def __init__(self, model_path: str, n_ctx: int, n_gpu_layers: int):
        try:
            from llama_cpp import Llama
        except Exception as e:
            raise RuntimeError("llama-cpp-python missing. Install with CUDA as discussed.") from e
        print(f"[green]Loading GGUF: {model_path}[/green]")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=True
        )

    def chat(self, user_text: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are concise and helpful."},
            {"role": "user",  "content": user_text},
        ]
        out = self.llm.create_chat_completion(messages=messages, max_tokens=256)
        return out["choices"][0]["message"]["content"].strip()


def make_llm():
    p = cfg["llm"]
    return LLMClient(p["model_path"], p["n_ctx"], p["n_gpu_layers"])
