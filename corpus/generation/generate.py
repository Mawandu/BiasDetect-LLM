"""Main script to generate texts from multiple LLMs for BiasDetect-LLM."""

import argparse
import logging
import sys
from typing import Any, Dict, List
from tqdm import tqdm

from config import (
    ANNOTATION_JSON_PATH,
    GENERATIONS_PER_PROMPT,
    MAX_TOKENS,
    MODELS,
    RAW_CSV_PATH,
    TEMPERATURE,
    ModelConfig,
    get_api_keys,
)
from prompts import PROMPTS
from utils import (
    LLMClientManager,
    format_item_id,
    get_utc_timestamp,
    load_existing_data,
    save_annotation_json,
    save_raw_generations_csv,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="BiasDetect-LLM Generation Pipeline")
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["openai", "anthropic", "groq", "mistral"],
        help="Optional: only run missing generations for a specific provider.",
    )
    return parser.parse_args()


def validate_api_keys(api_keys: Dict[str, str], models: List[ModelConfig]) -> List[str]:
    """Check which required API keys are missing for the target models."""
    required_keys = {m.env_key for m in models}
    missing = [key for key in sorted(required_keys) if not api_keys.get(key)]
    return missing


def run_generation_pipeline(only_provider: str | None = None) -> None:
    """Run or resume the text generation pipeline across all 24 prompts and 6 models (432 items)."""
    exact_map = load_existing_data(RAW_CSV_PATH)
    api_keys = get_api_keys()

    target_models = [
        m for m in MODELS if (only_provider is None or m.provider == only_provider)
    ]
    models_needing_keys = [
        m
        for m in target_models
        if not all(
            (p.prompt_id, m.model_name, g) in exact_map
            for p in PROMPTS
            for g in range(1, GENERATIONS_PER_PROMPT + 1)
        )
    ]
    missing_keys = validate_api_keys(api_keys, models_needing_keys)
    if missing_keys:
        print(
            f"[ERROR] Missing required API key(s) in .env: {', '.join(missing_keys)}\n"
            "Please add your API keys to the .env file before running generate.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    client_manager = LLMClientManager(api_keys=api_keys)

    total_tasks = len(PROMPTS) * len(MODELS) * GENERATIONS_PER_PROMPT
    records: List[Dict[str, Any]] = []
    preserved_count = 0
    skipped_other_provider_count = 0
    to_generate_count = 0

    task_plan: List[tuple[int, Any, ModelConfig, int, str, bool]] = []
    item_counter = 1

    for prompt_item in PROMPTS:
        for model_cfg in MODELS:
            for gen_idx in range(1, GENERATIONS_PER_PROMPT + 1):
                item_id = format_item_id(item_counter)
                existing_row = exact_map.get(
                    (prompt_item.prompt_id, model_cfg.model_name, gen_idx)
                )

                if existing_row is not None:
                    records.append(
                        {
                            "id": item_id,
                            "prompt_id": prompt_item.prompt_id,
                            "prompt_text": prompt_item.text,
                            "model": existing_row["model"],
                            "generation_index": gen_idx,
                            "text": existing_row["text"],
                            "timestamp": existing_row["timestamp"],
                        }
                    )
                    preserved_count += 1
                    should_run = False
                else:
                    records.append(
                        {
                            "id": item_id,
                            "prompt_id": prompt_item.prompt_id,
                            "prompt_text": prompt_item.text,
                            "model": model_cfg.model_name,
                            "generation_index": gen_idx,
                            "text": "",
                            "timestamp": "",
                        }
                    )
                    if only_provider is None or model_cfg.provider == only_provider:
                        to_generate_count += 1
                        should_run = True
                    else:
                        skipped_other_provider_count += 1
                        should_run = False

                task_plan.append(
                    (item_counter - 1, prompt_item, model_cfg, gen_idx, item_id, should_run)
                )
                item_counter += 1

    print(
        f"BiasDetect-LLM pipeline: {total_tasks} total items | "
        f"{preserved_count} already valid & preserved | "
        f"{to_generate_count} to generate."
    )

    if to_generate_count > 0:
        with tqdm(total=to_generate_count, desc="Generating texts", unit="gen") as pbar:
            for idx, prompt_item, model_cfg, gen_idx, item_id, should_run in task_plan:
                if not should_run:
                    continue

                pbar.set_postfix_str(
                    f"P{prompt_item.prompt_id:02d} | {model_cfg.provider}:{model_cfg.model_name} (#{gen_idx})"
                )

                actual_model = model_cfg.model_name
                try:
                    generated_text, actual_model = client_manager.generate_text(
                        provider=model_cfg.provider,
                        model_name=model_cfg.model_name,
                        prompt=prompt_item.text,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                except Exception as exc:
                    tqdm.write(
                        f"[ERROR] Failed generation for {item_id} "
                        f"(prompt={prompt_item.prompt_id}, model={model_cfg.model_name}, "
                        f"gen={gen_idx}): {exc}"
                    )
                    generated_text = f"[GENERATION_ERROR: {exc}]"

                records[idx] = {
                    "id": item_id,
                    "prompt_id": prompt_item.prompt_id,
                    "prompt_text": prompt_item.text,
                    "model": actual_model,
                    "generation_index": gen_idx,
                    "text": generated_text,
                    "timestamp": get_utc_timestamp(),
                }

                save_raw_generations_csv(records, RAW_CSV_PATH)
                pbar.update(1)

    save_raw_generations_csv(records, RAW_CSV_PATH)
    save_annotation_json(records, ANNOTATION_JSON_PATH)

    valid_total = sum(
        1
        for r in records
        if str(r.get("text", "")).strip()
        and not str(r.get("text", "")).startswith("[GENERATION_ERROR")
    )
    print(f"\nDataset ready: {valid_total}/{len(records)} valid items.")
    print(f"- Raw data saved to: {RAW_CSV_PATH}")
    print(f"- Annotation file saved to: {ANNOTATION_JSON_PATH}")


if __name__ == "__main__":
    args = parse_args()
    run_generation_pipeline(only_provider=args.provider)
