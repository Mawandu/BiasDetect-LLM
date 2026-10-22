"""Evaluation metrics and statistical tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def best_threshold(y, s) -> float:
    """Threshold maximizing macro-F1 (chosen on dev only)."""
    cands = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    return float(max(cands, key=lambda t: f1_score(y, s >= t, average="macro"))) if len(cands) else 0.5


def ece(y, p, bins=10) -> float:
    p = np.clip(p, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def metrics(y, s, thr) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    single = len(np.unique(y)) < 2
    return {
        "n": int(len(y)),
        "macro_f1": f1_score(y, s >= thr, average="macro"),
        "auroc": np.nan if single else roc_auc_score(y, s),
        "auprc": np.nan if single else average_precision_score(y, s),
        "ece": ece(y, s) if s.min() >= 0 and s.max() <= 1 else np.nan,
    }


def pairwise_accuracy(df: pd.DataFrame, s) -> float:
    """CrowS-Pairs style: fraction of pairs where the stereotypical sentence scores higher."""
    d = df.assign(s=s)
    piv = d.pivot_table(index="group_id", columns="label", values="s")
    piv = piv.dropna()
    return float(((piv[1] > piv[0]) + 0.5 * (piv[1] == piv[0])).mean())


def bootstrap_ci(y, s, thr, metric="macro_f1", n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metrics(y[i], s[i], thr)[metric])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar(y, pred_a, pred_b) -> float:
    """Exact McNemar test p-value between two systems' binary predictions."""
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
