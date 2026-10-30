"""Configuration module for the BiasDetect-LLM generation pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, List
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR: Path = Path(__file__).resolve().parent

# Load environment variables from .env or api_keys.env without overwriting non-empty keys
for env_filename in (".env", "api_keys.env"):
    env_path = BASE_DIR / env_filename
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

# Paths for raw data and final annotation output
DATA_DIR: Path = BASE_DIR / "data"
OUTPUT_DIR: Path = BASE_DIR / "output"

RAW_CSV_PATH: Path = DATA_DIR / "generated_raw.csv"
ANNOTATION_JSON_PATH: Path = OUTPUT_DIR / "texts_for_annotation.json"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation hyperparameters
TEMPERATURE: float = 0.8
MAX_TOKENS: int = 120
GENERATIONS_PER_PROMPT: int = 3

# Retry parameters for API calls
MAX_RETRIES: int = 4
INITIAL_BACKOFF_SECONDS: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    """Represents a target LLM configuration in the BiasDetect-LLM study."""
    provider: str
    model_name: str
    env_key: str
    slot_id: int = 0


# Final 6 LLMs evaluated in BiasDetect-LLM (24 prompts x 6 models x 3 generations = 432 texts)
MODELS: List[ModelConfig] = [
    ModelConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        env_key="OPENAI_API_KEY",
        slot_id=1,
    ),
    ModelConfig(
        provider="anthropic",
        model_name="claude-sonnet-4-6",
        env_key="ANTHROPIC_API_KEY",
        slot_id=2,
    ),
    ModelConfig(
        provider="groq",
        model_name="openai/gpt-oss-120b",
        env_key="GROQ_API_KEY",
        slot_id=3,
    ),
    ModelConfig(
        provider="groq",
        model_name="openai/gpt-oss-20b",
        env_key="GROQ_API_KEY",
        slot_id=4,
    ),
    ModelConfig(
        provider="groq",
        model_name="qwen/qwen3.8-27b",
        env_key="GROQ_API_KEY",
        slot_id=5,
    ),
    ModelConfig(
        provider="mistral",
        model_name="open-mistral-nemo",
        env_key="MISTRAL_API_KEY",
        slot_id=6,
    ),
]


def get_api_keys() -> Dict[str, str]:
    """Retrieve API keys from environment variables.

    Returns:
        Dict[str, str]: Mapping of environment variable names to their values.
    """
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "").strip(),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", "").strip(),
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY", "").strip(),
    }
