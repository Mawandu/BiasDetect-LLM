"""Provider-agnostic LLM client (OpenAI, Gemini, Groq) via OpenAI-compatible APIs.

API keys are read from environment variables (or a .env file):
    OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY

Model lists are editable in configs/llm.yaml so that new/deprecated
model names do not require code changes.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "llm.yaml"

PROVIDERS = {
    "openai": {"base_url": None, "env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env": "GEMINI_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env": "GROQ_API_KEY"},
}


def _load_dotenv():
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def available_models() -> dict[str, list[str]]:
    """Returns {provider: [models]} for providers whose API key is set."""
    _load_dotenv()
    cfg = load_config()["models"]
    return {p: cfg.get(p, []) for p, meta in PROVIDERS.items() if os.environ.get(meta["env"])}


@dataclass
class LLMClient:
    provider: str
    model: str
    temperature: float = 0.0
    max_retries: int = 5

    def __post_init__(self):
        from openai import OpenAI

        _load_dotenv()
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Choose among {list(PROVIDERS)}")
        meta = PROVIDERS[self.provider]
        key = os.environ.get(meta["env"])
        if not key:
            raise RuntimeError(f"Missing {meta['env']} (set it in the environment or in .env)")
        self._client = OpenAI(api_key=key, base_url=meta["base_url"])

    def chat(self, system: str, user: str, json_mode: bool = True) -> str:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except Exception as e:  # rate limits / transient errors
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"[{self.provider}] {type(e).__name__}: retry in {wait}s")
                time.sleep(wait)

    def chat_json(self, system: str, user: str) -> dict:
        raw = self.chat(system, user, json_mode=True)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Some models wrap JSON in markdown fences
            start, end = raw.find("{"), raw.rfind("}")
            return json.loads(raw[start:end + 1]) if start >= 0 else {}
