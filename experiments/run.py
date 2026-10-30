"""Main experiment runner.

Protocol: every system is trained on StereoSet-train, its threshold is tuned on
StereoSet-dev (macro-F1), then it is evaluated on
    - StereoSet-test (in-domain, unseen targets)
    - CrowS-Pairs    (out-of-domain, crowd-written pairs)
    - SBIC           (out-of-domain, real social-media posts)
Repeated over several split seeds. Scores are cached in results/preds/.

Usage
    python -m experiments.run                       # fast systems, seed 13
    python -m experiments.run --seeds 13 21 42 7 99 # 5 seeds
    python -m experiments.run --slow --llm --llm-limit 500
    python -m experiments.run --only P_fusion_minilm B1_legacy_rules
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from biasdetect.data.loaders import load_processed
from biasdetect.eval.metrics import best_threshold, metrics, pairwise_accuracy
from biasdetect.models import systems as S

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "results" / "preds"


def eval_sets(seed: int, llm_limit: int | None, is_llm: bool):
    ss = load_processed("stereoset", seed)
    sets = {"stereoset_test": ss[ss.split == "test"], "crows": load_processed("crows"),
            "sbic": load_processed("sbic")}
    if is_llm and llm_limit:  # paid API: stratified, reproducible subsample (pairs kept together)
        out = {}
        for k, d in sets.items():
            ids = pd.Series(d.pair_id.fillna(d.group_id) if "pair_id" in d else d.group_id).unique()
            rng = np.random.default_rng(0)
            keep = set(rng.choice(ids, size=min(len(ids), llm_limit // (2 if k != "sbic" else 1)), replace=False))
            key = d.pair_id.fillna(d.group_id) if "pair_id" in d else d.group_id
            out[k] = d[key.isin(keep)]
        sets = out
    return ss[ss.split == "train"], ss[ss.split == "dev"], sets


def run_system(sysobj, seed, llm_limit=None, no_cache=False):
    is_llm = sysobj.name.startswith("B5")
    train, dev, sets = eval_sets(seed, llm_limit, is_llm)
    sysobj.seed = seed
    t0 = time.time()
    sysobj.fit(train, dev)
    dev_s = sysobj.score(dev) if not is_llm else None
    thr = best_threshold(dev.label.values, dev_s) if dev_s is not None else 0.5
    rows = []
    for name, df in sets.items():
        path = PRED / f"{sysobj.name}__{name}__s{seed}.csv"
        cached = pd.read_csv(path) if path.exists() and not no_cache else None
        if cached is not None and (len(cached) != len(df) or len(cached) == 0
                                   or not {"score", "threshold"} <= set(cached.columns)):
            cached = None  # stale or incomplete cache: recompute
        if cached is not None:
            s = cached.score.values
            thr_used = float(cached.threshold.iloc[0])
        else:
            s = sysobj.score(df)
            thr_used = thr
            PRED.mkdir(parents=True, exist_ok=True)
            df.assign(score=s, threshold=thr_used).to_csv(path, index=False)
        m = metrics(df.label.values, s, thr_used)
        if "pair_id" in df and df.pair_id.notna().all():
            m["pair_acc"] = pairwise_accuracy(df.assign(group_id=df.pair_id), s)
        rows.append({"system": sysobj.name, "dataset": name, "seed": seed, "threshold": thr_used, **m})
    print(f"  {sysobj.name:<32} {time.time() - t0:6.1f}s")
    if isinstance(sysobj, S.HybridFusion):
        (ROOT / "results").mkdir(exist_ok=True)
        sysobj.branch_weights().to_csv(ROOT / "results" / f"weights_{sysobj.name}_s{seed}.csv")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[13])
    ap.add_argument("--slow", action="store_true", help="include zero-shot NLI (GPU recommended)")
    ap.add_argument("--llm", action="store_true", help="include LLM judge (paid API)")
    ap.add_argument("--llm-limit", type=int, default=400, help="max texts per dataset for the LLM judge")
    ap.add_argument("--finetuned", nargs="*", default=[], help="B4 model keys trained with train_transformer.py")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-cache", action="store_true", help="recompute predictions even if cached files exist")
    args = ap.parse_args()

    all_rows = []
    for seed in args.seeds:
        print(f"== seed {seed}")
        systems = S.registry(include_slow=args.slow, include_llm=args.llm)
        systems += [S.FineTuned(k) for k in args.finetuned]
        if args.only:
            systems = [s for s in systems if s.name in args.only]
        for sysobj in systems:
            try:
                all_rows += run_system(sysobj, seed, args.llm_limit, no_cache=args.no_cache)
            except Exception as e:
                print(f"  !! {sysobj.name} failed: {type(e).__name__}: {e}")
    res = pd.DataFrame(all_rows)
    out = ROOT / "results" / "metrics.csv"
    if out.exists():  # merge with previous runs (new rows override)
        old = pd.read_csv(out)
        res = pd.concat([old, res]).drop_duplicates(["system", "dataset", "seed"], keep="last")
    res.to_csv(out, index=False)
    show = res.groupby(["dataset", "system"])[["macro_f1", "auroc", "pair_acc"]].mean().round(3)
    print(show.to_string())


if __name__ == "__main__":
    main()
