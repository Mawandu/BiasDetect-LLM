"""Inter-annotator agreement + gold file.

    python -m biasdetect.annotation.agreement            # kappa + disagreements file
    python -m biasdetect.annotation.agreement --resolve  # after filling 'gold' in disagreements.csv

Outputs
    annotation/disagreements.csv   items to adjudicate (fill the 'gold' column)
    data/bold/gold.csv             final labels used for evaluation
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "annotation"


def load(a: str) -> pd.DataFrame:
    df = pd.read_csv(ANN / f"to_annotate_{a}.csv", dtype=str).fillna("")
    df["label"] = df.label.str.strip()
    bad = df[~df.label.isin(["0", "1", "?"])]
    if len(bad):
        raise ValueError(f"{a}: {len(bad)} items with invalid/missing label, e.g. {bad.item_id.head(3).tolist()}")
    return df.set_index("item_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="A1")
    ap.add_argument("--b", default="A2")
    ap.add_argument("--resolve", action="store_true")
    args = ap.parse_args()

    A, B = load(args.a), load(args.b)
    m = A[["label", "category"]].join(B[["label", "category"]], lsuffix="_a", rsuffix="_b")
    both = m[(m.label_a != "?") & (m.label_b != "?")]
    kappa = cohen_kappa_score(both.label_a, both.label_b)
    print(f"Items: {len(m)} | both decided: {len(both)} | raw agreement: {(both.label_a == both.label_b).mean():.3f}")
    print(f"Cohen's kappa (label): {kappa:.3f}")
    pos = both[(both.label_a == "1") & (both.label_b == "1")]
    if len(pos) > 1:
        print(f"Cohen's kappa (category | both=1): {cohen_kappa_score(pos.category_a, pos.category_b):.3f}")
    master = pd.read_csv(ANN / "items_master.csv").set_index("item_id")
    m = m.join(master[["domain"]])
    print("\nKappa per domain:")
    for d, sub in m[(m.label_a != "?") & (m.label_b != "?")].groupby("domain"):
        k = cohen_kappa_score(sub.label_a, sub.label_b) if sub.label_a.nunique() + sub.label_b.nunique() > 2 else float("nan")
        print(f"  {d:<20} n={len(sub):<4} kappa={k:.3f}")

    dis_path = ANN / "disagreements.csv"
    agree = m[(m.label_a == m.label_b) & (m.label_a != "?")]
    if not args.resolve:
        dis = m.drop(agree.index).join(A[["text"]]).assign(gold="")
        dis.to_csv(dis_path)
        print(f"\n{len(dis)} items to adjudicate -> {dis_path} (fill 'gold' with 0/1, then rerun with --resolve)")
        return
    dis = pd.read_csv(dis_path, dtype=str).set_index("item_id")
    gold = pd.concat([agree.label_a.rename("gold"), dis.gold[dis.gold.isin(["0", "1"])]])
    out = master.join(gold, how="inner").assign(label=lambda d: d.gold.astype(int))
    out_path = ROOT / "data" / "bold" / "gold.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path)
    print(f"gold.csv: {len(out)} items, {out.label.mean():.1%} stereotypical")


if __name__ == "__main__":
    main()
