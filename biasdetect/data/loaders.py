"""Dataset loaders. Every loader returns a DataFrame with a common schema:

    text      : str   sentence / post to classify
    label     : int   1 = stereotypical, 0 = not stereotypical
    category  : str   gender | origin | religion | profession
    group_id  : str   unit used for leakage-free splitting (target / pair)
    source    : str   dataset name
"""
from __future__ import annotations

import ast
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

CATEGORIES = ["gender", "origin", "religion", "profession"]

URLS = {
    "crows": "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv",
    "sbic": "https://homes.cs.washington.edu/~msap/social-bias-frames/SBIC.v2.tgz",
}


def _download(url: str, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, dest)


# ---------------------------------------------------------------- StereoSet
STEREOSET_MAP = {"gender": "gender", "race": "origin", "religion": "religion", "profession": "profession"}


def load_stereoset(include_unrelated: bool = False) -> pd.DataFrame:
    """StereoSet intrasentence (Nadeem et al., 2021). Only the dev split is public.

    Main task: stereotype (1) vs anti-stereotype (0). Both sentences share the same
    context and differ by a single word, which makes this a hard, controlled task.
    'unrelated' sentences are nonsensical and are excluded by default.
    """
    from datasets import load_dataset

    ds = load_dataset("McGill-NLP/stereoset", "intrasentence")["validation"]
    names = ds.features["sentences"]["gold_label"].feature.names
    rows = []
    for ex in ds:
        for sent, gold in zip(ex["sentences"]["sentence"], ex["sentences"]["gold_label"]):
            g = names[gold]
            if g == "unrelated" and not include_unrelated:
                continue
            rows.append({
                "text": sent,
                "label": int(g == "stereotype"),
                "category": STEREOSET_MAP[ex["bias_type"]],
                "group_id": f"ss::{ex['target'].lower()}",
                "pair_id": f"ss::{ex['id']}",
                "target": ex["target"],
                "source": "stereoset",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CrowS-Pairs
CROWS_MAP = {"gender": "gender", "race-color": "origin", "nationality": "origin", "religion": "religion"}


def load_crows() -> pd.DataFrame:
    """CrowS-Pairs (Nangia et al., 2020), restricted to our categories.

    'sent_more' is always about the more disadvantaged group; for 'stereo' rows it is
    the stereotypical sentence, for 'antistereo' rows the stereotypical one is 'sent_less'.
    """
    path = RAW / "crows_pairs.csv"
    _download(URLS["crows"], path)
    df = pd.read_csv(path)
    df = df[df.bias_type.isin(CROWS_MAP)]
    rows = []
    for i, r in df.iterrows():
        stereo, anti = (r.sent_more, r.sent_less) if r.stereo_antistereo == "stereo" else (r.sent_less, r.sent_more)
        for text, label in ((stereo, 1), (anti, 0)):
            rows.append({"text": text, "label": label, "category": CROWS_MAP[r.bias_type],
                         "group_id": f"cp::{i}", "pair_id": f"cp::{i}", "source": "crows"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- SBIC
def _sbic_category(cats: list[str]) -> str | None:
    cats = set(cats)
    if "gender" in cats:
        return "gender"
    if "race" in cats or "culture" in cats:
        return "origin"
    if "religion" in cats:  # not a native SBIC category; kept for robustness
        return "religion"
    return None


def load_sbic(split: str = "tst") -> pd.DataFrame:
    """Social Bias Frames (Sap et al., 2020), aggregated file.

    label = 1 if annotators wrote at least one implied stereotype (targetStereotype
    non-empty). NB: the 'hasBiasedImplication' column is inverted in the release
    (1 = no implication), hence we do not use it.
    Positives are restricted to gender / race / culture targets; negatives are posts
    with no implied stereotype.
    """
    path = RAW / f"SBIC.v2.agg.{split}.csv"
    if not path.exists():
        tgz = RAW / "SBIC.v2.tgz"
        _download(URLS["sbic"], tgz)
        with tarfile.open(tgz) as t:
            t.extractall(RAW)
    df = pd.read_csv(path)
    rows = []
    for i, r in df.iterrows():
        stereos = ast.literal_eval(r.targetStereotype)
        cats = ast.literal_eval(r.targetCategory)
        text = " ".join(str(r.post).split())
        if stereos:
            cat = _sbic_category(cats)
            if cat is None:
                continue
            rows.append({"text": text, "label": 1, "category": cat, "group_id": f"sbic::{i}", "source": "sbic"})
        else:
            rows.append({"text": text, "label": 0, "category": "none", "group_id": f"sbic::{i}", "source": "sbic"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Splits
def group_split(df: pd.DataFrame, ratios=(0.6, 0.2, 0.2), seed: int = 13) -> pd.DataFrame:
    """Leakage-free split: all rows sharing a group_id (e.g. a StereoSet target)
    go to the same split. Stratified by category."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = ""
    for cat, sub in df.groupby("category"):
        groups = np.array(sub.group_id.unique())
        rng.shuffle(groups)
        n = len(groups)
        n_tr, n_dv = int(ratios[0] * n), int(ratios[1] * n)
        assign = {g: "train" for g in groups[:n_tr]}
        assign.update({g: "dev" for g in groups[n_tr:n_tr + n_dv]})
        assign.update({g: "test" for g in groups[n_tr + n_dv:]})
        out.loc[sub.index, "split"] = sub.group_id.map(assign)
    return out


def build_all(seed: int = 13) -> dict[str, pd.DataFrame]:
    """Builds and caches all evaluation sets in data/processed/.
    StereoSet is split per seed (stereoset_s{seed}.csv)."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    sets = {f"stereoset_s{seed}": group_split(load_stereoset(), seed=seed),
            "crows": load_crows(), "sbic": load_sbic("tst")}
    for name, df in sets.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
    return sets


def load_processed(name: str, seed: int = 13) -> pd.DataFrame:
    """name in {'stereoset', 'crows', 'sbic'}; StereoSet split depends on seed."""
    fname = f"stereoset_s{seed}" if name == "stereoset" else name
    path = PROCESSED / f"{fname}.csv"
    if not path.exists():
        build_all(seed)
    return pd.read_csv(path)


if __name__ == "__main__":
    for name, df in build_all().items():
        print(f"\n== {name}: {len(df)} rows")
        cols = ["split", "category"] if "split" in df else ["category"]
        print(df.groupby(cols).label.agg(["count", "mean"]).round(3))
