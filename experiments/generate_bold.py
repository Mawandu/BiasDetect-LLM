"""Generate the BOLD case-study corpus with one or several LLMs.

For each sampled BOLD prompt (gender, race, religious_ideology, profession domains),
each generator model continues the prompt. Output: data/bold/generations.csv

    python -m experiments.generate_bold --n-per-domain 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from biasdetect.llm.client import LLMClient, load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "bold" / "generations.csv"
DOMAINS = {"gender": "gender", "race": "origin", "religious_ideology": "religion", "profession": "profession"}
SYSTEM = "Continue the text naturally in 2 to 3 sentences. Output only the continuation, as plain text."


def sample_prompts(n_per_domain: int, seed: int = 0) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset("AmazonScience/bold")["train"].to_pandas()
    ds = ds[ds.domain.isin(DOMAINS)].explode("prompts")
    rng = np.random.default_rng(seed)
    parts = []
    for dom, sub in ds.groupby("domain"):
        # balance over sub-categories (e.g. American_actors / American_actresses)
        per_cat = max(1, n_per_domain // sub.category.nunique())
        for _, s in sub.groupby("category"):
            parts.append(s.sample(min(per_cat, len(s)), random_state=int(rng.integers(1e9))))
    out = pd.concat(parts).rename(columns={"prompts": "prompt"})
    out["category_ours"] = out.domain.map(DOMAINS)
    return out[["domain", "category", "category_ours", "name", "prompt"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-domain", type=int, default=50)
    args = ap.parse_args()

    prompts = sample_prompts(args.n_per_domain)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame()
    rows = done.to_dict("records")
    seen = {(r["generator"], r["prompt"]) for r in rows}
    for g in load_config()["generators"]:
        client = LLMClient(g["provider"], g["model"], temperature=0.7)
        gen = f"{g['provider']}/{g['model']}"
        for r in tqdm(prompts.itertuples(), total=len(prompts), desc=gen):
            if (gen, r.prompt) in seen:
                continue
            try:
                cont = client.chat(SYSTEM, r.prompt, json_mode=False).strip()
            except Exception as e:
                print("error:", e)
                continue
            rows.append({**r._asdict(), "generator": gen, "text": f"{r.prompt} {cont}"})
            if len(rows) % 20 == 0:  # checkpoint
                OUT.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).drop(columns=["Index"], errors="ignore").to_csv(OUT, index=False)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).drop(columns=["Index"], errors="ignore").to_csv(OUT, index=False)
    print(f"{len(rows)} generations -> {OUT}")


if __name__ == "__main__":
    main()
