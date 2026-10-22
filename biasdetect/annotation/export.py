"""Export annotation sheets (same items, different order per annotator, no system outputs).

    python -m biasdetect.annotation.export --n 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "bold" / "generations.csv"
OUT = ROOT / "annotation"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()

    df = pd.read_csv(SRC)
    # stratified sample over domain x generator
    n_cells = df.groupby(["domain", "generator"]).ngroups
    per = max(1, args.n // n_cells)
    items = (df.groupby(["domain", "generator"], group_keys=False)
               .apply(lambda g: g.sample(min(per, len(g)), random_state=0))
               .reset_index(drop=True))
    items.insert(0, "item_id", [f"it{i:04d}" for i in range(len(items))])
    OUT.mkdir(exist_ok=True)
    items.to_csv(OUT / "items_master.csv", index=False)  # keeps metadata, NOT given to annotators

    for k, a in enumerate(args.annotators):
        sheet = items[["item_id", "text"]].sample(frac=1, random_state=100 + k).assign(label="", category="", comment="")
        sheet.to_csv(OUT / f"to_annotate_{a}.csv", index=False)
    print(f"{len(items)} items -> {OUT}/to_annotate_*.csv (guide: docs/annotation_guide.md)")


if __name__ == "__main__":
    main()
