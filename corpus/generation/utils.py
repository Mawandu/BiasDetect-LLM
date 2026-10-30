"""Utility functions for API calls, retry handling, and file persistence."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List

from anthropic import Anthropic
from groq import Groq
from openai import OpenAI
import pandas as pd

from config import INITIAL_BACKOFF_SECONDS, MAX_RETRIES

logger = logging.getLogger(__name__)

RAW_CSV_COLUMNS: List[str] = [
    "id",
    "prompt_id",
    "prompt_text",
    "model",
    "generation_index",
    "text",
    "timestamp",
]


def get_utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def format_item_id(index: int) -> str:
    """Format a sequential integer index as 'item_XXX' (e.g., 'item_001')."""
    return f"item_{index:03d}"


def clean_generated_text(text: str) -> str:
    """Strip <think>...</think> reasoning blocks and surrounding whitespace."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()


def retry_with_backoff(
    func: Callable[[], str],
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF_SECONDS,
) -> str:
    """Execute a callable with exponential backoff on transient errors or rate limits.

    Does not retry on non-retryable client errors (400, 401, 403, 404).
    """
    backoff = initial_backoff
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exception = exc
            status_code = getattr(exc, "status_code", None)
            if isinstance(exc, (TypeError, ValueError)) or status_code in (400, 401, 403, 404):
                break
            if attempt == max_retries:
                logger.error("Failed after %d attempts: %s", max_retries, exc)
                break
            logger.warning(
                "API error on attempt %d/%d (%s). Retrying in %.1fs...",
                attempt,
                max_retries,
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff *= 2.0

    assert last_exception is not None
    raise last_exception


class LLMClientManager:
    """Manages official SDK clients and text generation across OpenAI, Anthropic, Groq, and Mistral."""

    def __init__(self, api_keys: Dict[str, str]) -> None:
        self.api_keys = api_keys
        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None
        self._groq_client: Groq | None = None
        self._mistral_client: OpenAI | None = None

    @property
    def openai_client(self) -> OpenAI:
        if self._openai_client is None:
            key = self.api_keys.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("Missing OPENAI_API_KEY in environment / .env file.")
            self._openai_client = OpenAI(api_key=key)
        return self._openai_client

    @property
    def anthropic_client(self) -> Anthropic:
        if self._anthropic_client is None:
            key = self.api_keys.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError("Missing ANTHROPIC_API_KEY in environment / .env file.")
            self._anthropic_client = Anthropic(api_key=key)
        return self._anthropic_client

    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            key = self.api_keys.get("GROQ_API_KEY")
            if not key:
                raise ValueError("Missing GROQ_API_KEY in environment / .env file.")
            self._groq_client = Groq(api_key=key)
        return self._groq_client

    @property
    def mistral_client(self) -> OpenAI:
        if self._mistral_client is None:
            key = self.api_keys.get("MISTRAL_API_KEY")
            if not key:
                raise ValueError("Missing MISTRAL_API_KEY in environment / .env file.")
            self._mistral_client = OpenAI(
                api_key=key,
                base_url="https://api.mistral.ai/v1",
                max_retries=0,
            )
        return self._mistral_client

    def _call_single_model(
        self,
        provider: str,
        target_model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Perform a single API request for a specific provider and model."""
        if provider == "openai":
            response = self.openai_client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            text = clean_generated_text(content or "")
            if not text:
                raise RuntimeError(f"Empty response from OpenAI model {target_model}")
            return text

        if provider == "anthropic":
            response = self.anthropic_client.messages.create(
                model=target_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_parts: List[str] = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            text = clean_generated_text("\n".join(text_parts))
            if not text:
                raise RuntimeError(f"Empty response from Anthropic model {target_model}")
            return text

        if provider == "groq":
            is_reasoning_model = any(
                prefix in target_model.lower()
                for prefix in ("gpt-oss", "qwen", "deepseek", "r1")
            )
            extra_kwargs: Dict[str, Any] = {}
            if "gpt-oss" in target_model.lower():
                extra_kwargs["reasoning_effort"] = "low"
                extra_kwargs["include_reasoning"] = False
            elif "qwen" in target_model.lower():
                extra_kwargs["reasoning_effort"] = "none"

            effective_max_tokens = 512 if is_reasoning_model else max_tokens

            try:
                response = self.groq_client.chat.completions.create(
                    model=target_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    **extra_kwargs,
                )
            except Exception as exc:
                if getattr(exc, "status_code", None) == 400 and extra_kwargs:
                    response = self.groq_client.chat.completions.create(
                        model=target_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=effective_max_tokens,
                    )
                else:
                    raise

            content = response.choices[0].message.content
            text = clean_generated_text(content or "")
            if not text:
                raise RuntimeError(f"Empty response from Groq model {target_model}")
            return text

        if provider == "mistral":
            # Respect Mistral Free Tier 1 req/sec rate limit
            time.sleep(1.5)
            response = self.mistral_client.chat.completions.create(
                model=target_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            text = clean_generated_text(content or "")
            if not text:
                raise RuntimeError(f"Empty response from Mistral model {target_model}")
            return text

        raise ValueError(f"Unsupported provider: {provider}")

    def generate_text(
        self,
        provider: str,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, str]:
        """Generate a completion for the specified provider and model with retry support.

        Returns:
            tuple[str, str]: (generated_text, actual_model_used)
        """
        text = retry_with_backoff(
            lambda: self._call_single_model(
                provider=provider,
                target_model=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return text, model_name


def load_existing_data(csv_path: Path) -> Dict[tuple[int, str, int], Dict[str, Any]]:
    """Load existing valid rows from generated_raw.csv keyed by (prompt_id, model, generation_index)."""
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        exact_map: Dict[tuple[int, str, int], Dict[str, Any]] = {}
        for row in df.to_dict(orient="records"):
            text = str(row.get("text", "")).strip()
            if not text or text.startswith("[GENERATION_ERROR"):
                continue
            p_id = int(row["prompt_id"])
            m_name = str(row["model"]).strip()
            g_idx = int(row["generation_index"])
            exact_map[(p_id, m_name, g_idx)] = row
        return exact_map
    except Exception as exc:
        logger.warning("Could not read existing CSV (%s): %s", csv_path, exc)
        return {}


def save_raw_generations_csv(records: List[Dict[str, Any]], csv_path: Path) -> None:
    """Save all raw generation records to a CSV file using pandas."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records, columns=RAW_CSV_COLUMNS)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def save_annotation_json(records: List[Dict[str, Any]], json_path: Path) -> None:
    """Save anonymized records (only 'id' and 'text') to a JSON file for Potato annotation."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    annotation_items: List[Dict[str, str]] = [
        {"id": str(record["id"]), "text": str(record["text"])}
        for record in records
    ]
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(annotation_items, f, indent=2, ensure_ascii=False)
