"""Builds the BiasDetect ontology (RDF/Turtle) from citable sources.

Classes
    bd:SocialGroupTerm   a lexical term denoting a social group  (category, pole)
    bd:Occupation        an occupation with a measured gender skew (BLS)
    bd:StereoAttribute   a word statistically associated with stereotypes of a category
Every individual carries `prov:wasDerivedFrom` pointing to its source.

Sources
    S1  Zhao et al. (2018) GN-GloVe gendered word lists               -> gender terms (male/female)
    S2  HolisticBias descriptors, Smith et al. (2022)                  -> origin / religion / gender terms,
                                                                          + 'dispreferred' flag
    S3  BLS % female per occupation via Winogender, Rudinger et al. (2018)
    S4  StereoSet *train split only* (Nadeem et al., 2021)             -> attribute words with log-odds
                                                                          of appearing in stereotype vs
                                                                          anti-stereotype sentences
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from rdflib.namespace import PROV, XSD

from biasdetect.data.loaders import load_processed

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "ontology"
ONT_DIR = ROOT / "data" / "ontology"


def ttl_path(seed: int = 13) -> Path:
    return ONT_DIR / f"bias_s{seed}.ttl"

BD = Namespace("http://biasdetect.org/ontology#")
SOURCES = {
    "S1": ("Zhao et al. 2018, GN-GloVe gendered word lists", "https://github.com/uclanlp/gn_glove"),
    "S2": ("Smith et al. 2022, HolisticBias descriptors v1.0", "https://github.com/facebookresearch/ResponsibleNLP"),
    "S3": ("BLS % female (2015) via Rudinger et al. 2018, Winogender", "https://github.com/rudinger/winogender-schemas"),
    "S4": ("StereoSet intrasentence, train split only (Nadeem et al. 2021)", "https://huggingface.co/datasets/McGill-NLP/stereoset"),
}
HB_MAP = {"race_ethnicity": "origin", "nationality": "origin", "cultural": "origin",
          "religion": "religion", "gender_and_sex": "gender"}

TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")
STOP = set("""a an the is was were be been are am of to in on at for and or but with by as that this
it its he she his her him they them their we our you your i my me not no so very too from
who whom which what when where why how all any some one just than then there here also into
""".split())


def _uri(kind: str, term: str) -> URIRef:
    return BD[f"{kind}_{re.sub(r'[^a-zA-Z0-9]+', '_', term.strip().lower())}"]


def _hb_descriptors(node):
    if isinstance(node, dict):
        for v in node.values():
            yield from _hb_descriptors(v)
    elif isinstance(node, list):
        for x in node:
            if isinstance(x, str):
                yield x, "reviewed"
            elif isinstance(x, dict):
                yield x["descriptor"], x.get("preference", "reviewed")


def stereo_attributes(seed: int = 13, min_count: int = 3, alpha: float = 1.0) -> pd.DataFrame:
    """Log-odds (with add-alpha smoothing) of a word occurring in stereotype vs
    anti-stereotype sentences of the StereoSet *train* split, per category."""
    df = load_processed("stereoset", seed)
    df = df[df.split == "train"]
    rows = []
    for cat, sub in df.groupby("category"):
        cnt = {1: Counter(), 0: Counter()}
        for text, lab in zip(sub.text, sub.label):
            cnt[lab].update(set(t for t in TOKEN_RE.findall(text.lower()) if t not in STOP))
        n1, n0 = (sub.label == 1).sum(), (sub.label == 0).sum()
        for w in set(cnt[1]) | set(cnt[0]):
            c1, c0 = cnt[1][w], cnt[0][w]
            if c1 + c0 < min_count:
                continue
            lo = math.log((c1 + alpha) / (n1 - c1 + alpha)) - math.log((c0 + alpha) / (n0 - c0 + alpha))
            rows.append({"word": w, "category": cat, "log_odds": lo, "n_stereo": c1, "n_anti": c0})
    return pd.DataFrame(rows)


def build(seed: int = 13, min_abs_log_odds: float = 0.5) -> Graph:
    g = Graph()
    g.bind("bd", BD)
    g.bind("prov", PROV)

    for cls in ("SocialGroupTerm", "Occupation", "StereoAttribute", "Source"):
        g.add((BD[cls], RDF.type, RDFS.Class))
    for sid, (label, url) in SOURCES.items():
        g.add((BD[sid], RDF.type, BD.Source))
        g.add((BD[sid], RDFS.label, Literal(label)))
        g.add((BD[sid], RDFS.seeAlso, URIRef(url)))

    def add_term(term, category, pole, source, preference=None):
        u = _uri("term", term)
        g.add((u, RDF.type, BD.SocialGroupTerm))
        g.add((u, RDFS.label, Literal(term.lower())))
        g.add((u, BD.category, Literal(category)))
        if pole:
            g.add((u, BD.pole, Literal(pole)))
        if preference:
            g.add((u, BD.preference, Literal(preference)))
        g.add((u, PROV.wasDerivedFrom, BD[source]))

    # S1 gendered words
    for fname, pole in (("male_word_file.txt", "male"), ("female_word_file.txt", "female")):
        for w in (SRC / fname).read_text().split():
            add_term(w, "gender", pole, "S1")

    # S2 HolisticBias
    hb = json.loads((SRC / "descriptors.json").read_text())
    for axis, cat in HB_MAP.items():
        for desc, pref in _hb_descriptors(hb.get(axis, {})):
            add_term(desc, cat, None, "S2", pref)

    # S3 occupations
    occ = pd.read_csv(SRC / "occupations-stats.tsv", sep="\t")
    for r in occ.itertuples():
        u = _uri("occ", r.occupation)
        g.add((u, RDF.type, BD.Occupation))
        g.add((u, RDFS.label, Literal(r.occupation.lower())))
        g.add((u, BD.pctFemale, Literal(float(r.bls_pct_female), datatype=XSD.decimal)))
        g.add((u, BD.statYear, Literal(int(r.bls_year))))
        g.add((u, PROV.wasDerivedFrom, BD.S3))

    # S4 stereotype attributes (train split only)
    attrs = stereo_attributes(seed)
    attrs = attrs[attrs.log_odds.abs() >= min_abs_log_odds]
    for r in attrs.itertuples():
        u = _uri(f"attr_{r.category}", r.word)
        g.add((u, RDF.type, BD.StereoAttribute))
        g.add((u, RDFS.label, Literal(r.word)))
        g.add((u, BD.category, Literal(r.category)))
        g.add((u, BD.logOdds, Literal(round(float(r.log_odds), 4), datatype=XSD.decimal)))
        g.add((u, BD.nStereo, Literal(int(r.n_stereo))))
        g.add((u, BD.nAnti, Literal(int(r.n_anti))))
        g.add((u, PROV.wasDerivedFrom, BD.S4))

    ONT_DIR.mkdir(parents=True, exist_ok=True)
    g.serialize(ttl_path(seed), format="turtle")
    return g


if __name__ == "__main__":
    import sys
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    g = build(seed)
    for cls in ("SocialGroupTerm", "Occupation", "StereoAttribute"):
        print(cls, len(set(g.subjects(RDF.type, BD[cls]))))
    print(f"-> {ttl_path(seed)} ({len(g)} triples)")
