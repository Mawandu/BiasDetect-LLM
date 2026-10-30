#!/usr/bin/env python
"""Construit corpus/ (versionné) à partir du projet d'annotation.
Usage : python scripts/build_corpus.py --src ~/biasdetect_annotation --out corpus
Ne modifie rien dans --src. Refuse d'écraser --out. N'inclut ni commentaires, ni pilote, ni noms d'annotateurs.
"""
import argparse, datetime, hashlib, platform, sys
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd, sklearn
from sklearn.metrics import cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="~/biasdetect_annotation")
ap.add_argument("--out", default="corpus")
a = ap.parse_args()
src, out = Path(a.src).expanduser(), Path(a.out)
if out.exists() and any(out.iterdir()):
    sys.exit(f"{out} existe déjà et n'est pas vide : ne pas écraser")
out.mkdir(parents=True, exist_ok=True)

# ---- 1er tour (étiquettes brutes) ----
raw = pd.read_csv(src/"annotation_output/exports/csv/annotations.csv")
raw = raw[raw.instance_id.str.startswith("item_") & raw.user_id.isin(["A1","A2","A3"])].copy()
raw["y"] = np.where(raw["label.?"].notna(), "?", np.where(raw["label.1"].notna(), "1", "0"))
assert len(raw) == 1296 and not raw.duplicated(["instance_id","user_id"]).any(), "export inattendu"
w = raw.pivot(index="instance_id", columns="user_id", values="y")
assert w.shape == (432, 3) and w.notna().all().all()

def maj(r):
    v = [int(x) for x in r if x != "?"]; s = sum(v)
    if len(v) == 3: return int(s >= 2)
    if len(v) == 2 and s in (0, 2): return s // 2
    return np.nan                                  # égalité : à adjuger
gold = w.apply(maj, axis=1)
if gold.isna().any():
    sys.exit(f"égalités à adjuger : {gold[gold.isna()].index.tolist()}")

# ---- métadonnées + textes ----
d = pd.read_csv(src/"dataset_final_annotated.csv")[["id","prompt_id","prompt_text","model","generation_index","text","prompt_category"]]
t = d["text"].fillna("").str.strip(); nw = t.str.split().str.len()
d["n_words"] = nw
d["refusal"] = t.str.contains(r"(?i)can[’']?t help|cannot help|can[’']?t assist|cannot assist|can[’']?t continue|I[’']m sorry", regex=True) & (nw < 25)
d["is_core"] = d.prompt_category != "OTHER"
c = d.set_index("id").join(w.rename(columns=lambda x: f"{x}_round1"))
c["n_valid_round1"] = (w != "?").sum(axis=1)
c["gold_raw"] = gold.astype(int)
rev = pd.read_csv(src/"gold_final_v2.csv").set_index("instance_id")["gold"]
c["gold_reviewed"] = rev.astype(int)
c["changed_by_review"] = c.gold_raw != c.gold_reviewed
c = c.reset_index()
cols = ["id","prompt_id","prompt_category","is_core","model","generation_index","text","n_words","refusal",
        "A1_round1","A2_round1","A3_round1","n_valid_round1","gold_raw","gold_reviewed","changed_by_review","prompt_text"]
c[cols].to_csv(out/"biasdetect_corpus.csv", index=False)

# ---- accord (1er tour, items à 3 étiquettes valides) ----
def fleiss(x):
    n = 3; n1 = x.sum(axis=1).values; n0 = n - n1
    Pb = ((n0*(n0-1) + n1*(n1-1)) / (n*(n-1))).mean(); p1 = n1.sum() / (len(x)*n)
    Pe = p1**2 + (1-p1)**2
    return Pb, ((Pb-Pe)/(1-Pe) if Pe < 1 else np.nan), ((Pb-2*p1*(1-p1))/(1-2*p1*(1-p1)) if 0 < p1 < 1 else np.nan)
v = w[(w != "?").all(axis=1)].astype(int)
meta = c.set_index("id")
rng = np.random.default_rng(13)
def row(name, x):
    pid = meta.loc[x.index, "prompt_id"].values; up = np.unique(pid)
    idx = {p: np.where(pid == p)[0] for p in up}; bs = []
    for _ in range(1000):
        s = np.concatenate([idx[p] for p in rng.choice(up, len(up))]); bs.append(fleiss(x.iloc[s])[1])
    po, k, g = fleiss(x)
    return dict(subset=name, N=len(x), observed_agreement=round(po,3), fleiss_kappa=round(k,3),
                kappa_ci95_low=round(np.nanpercentile(bs,2.5),3), kappa_ci95_high=round(np.nanpercentile(bs,97.5),3), gwet_ac1=round(g,3))
cat = meta.loc[v.index, "prompt_category"]; ref = meta.loc[v.index, "refusal"]
rows = [row("all", v)] + [row(k, v[cat == k]) for k in sorted(cat.unique())]
rows += [row("core (all except OTHER)", v[cat != "OTHER"]), row("all excluding refusals", v[~ref])]
for x, y in combinations(v.columns, 2):
    rows.append(dict(subset=f"pair {x}-{y} (cohen)", N=len(v), observed_agreement=round((v[x]==v[y]).mean(),3),
                     fleiss_kappa=round(cohen_kappa_score(v[x], v[y]),3)))
summ = pd.DataFrame(rows); summ.to_csv(out/"agreement_summary_round1.csv", index=False)
print(summ.to_string(index=False))
print("\ngold_raw :", c.gold_raw.value_counts().to_dict(), "| gold_reviewed :", c.gold_reviewed.value_counts().to_dict(),
      "| changés :", int(c.changed_by_review.sum()), "| refus :", int(c.refusal.sum()))

(out/"README.md").write_text(f"""# BiasDetect-Corpus v1
Built {datetime.date.today()} with Python {platform.python_version()}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}.
- biasdetect_corpus.csv: 432 texts (24 prompts x 6 models x 3 samples) with round-1 labels of annotators A1-A3 (1/0/?),
  gold_raw = round-1 majority vote (PRIMARY), gold_reviewed = after the second round on 44 items (sensitivity only),
  refusal flag (regex, <25 words), is_core (GENDER/ORIGIN/RELIGION).
- agreement_summary_round1.csv: Fleiss kappa (prompt-level bootstrap CI), Gwet AC1, pairwise Cohen kappa, on items with 3 valid labels.
- Annotator names, comments, pilot items and annotation logs are not included.
- Known limits: 24 prompts; correlated samples; token limits differed between models; no guide rule for refusals.
""")
with open(out/"SHA256SUMS", "w") as f:
    for p in sorted(out.iterdir()):
        if p.name != "SHA256SUMS": f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
print("OK :", sorted(p.name for p in out.iterdir()))
