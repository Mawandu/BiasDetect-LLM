"""Symbolic features from the ontology + explanations (rules fired, with provenance)."""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdflib import Graph, RDF
from rdflib.namespace import PROV, RDFS
from sklearn.model_selection import GroupKFold

from biasdetect.ontology.build import BD, SOURCES, build, stereo_attributes_from, ttl_path

CATS = ["gender", "origin", "religion", "profession"]
TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")
FEMALE_PRON = {"she", "her", "hers", "herself"}
MALE_PRON = {"he", "him", "his", "himself"}
MIN_ABS_LOG_ODDS = 0.5


def _attributes_index(attributes: pd.DataFrame) -> dict[tuple[str, str], float]:
    index = {}
    for r in attributes.itertuples(index=False):
        lo = float(r.log_odds)
        if abs(lo) >= MIN_ABS_LOG_ODDS:
            index[(str(r.word), str(r.category))] = lo
    return index


@dataclass
class Rule:
    rule: str
    term: str
    detail: str
    source: str


@dataclass
class SymbolicResult:
    features: dict
    rules: list = field(default_factory=list)


class Ontology:
    """In-memory index of the RDF ontology for fast lexical matching."""

    def __init__(self, seed: int = 13, attributes: pd.DataFrame | None = None):
        path = ttl_path(seed)
        if not Path(path).exists():
            build(seed)
        g = Graph().parse(path, format="turtle")
        self.group_terms: dict[str, dict] = {}
        self.max_ngram = 1
        for s in g.subjects(RDF.type, BD.SocialGroupTerm):
            label = str(g.value(s, RDFS.label))
            self.group_terms[label] = {
                "category": str(g.value(s, BD.category)),
                "pole": str(g.value(s, BD.pole) or ""),
                "dispreferred": str(g.value(s, BD.preference) or "") == "dispreferred",
                "source": str(g.value(s, PROV.wasDerivedFrom)).split("#")[-1],
            }
            self.max_ngram = max(self.max_ngram, len(label.split()))
        self.occupations = {
            str(g.value(s, RDFS.label)): float(g.value(s, BD.pctFemale))
            for s in g.subjects(RDF.type, BD.Occupation)
        }
        self.attributes: dict[tuple[str, str], float] = {}
        if attributes is None:
            for s in g.subjects(RDF.type, BD.StereoAttribute):
                self.attributes[(str(g.value(s, RDFS.label)), str(g.value(s, BD.category)))] = float(g.value(s, BD.logOdds))
        else:
            self.attributes = _attributes_index(attributes)

    def source_label(self, sid: str) -> str:
        return SOURCES.get(sid, (sid,))[0]


class SymbolicFeaturizer:
    FEATURE_NAMES = (
        [f"grp_{c}" for c in CATS[:3]]
        + ["grp_dispreferred", "occ_present", "occ_skew", "occ_congruent", "occ_incongruent"]
        + [f"attr_max_{c}" for c in CATS] + [f"attr_min_{c}" for c in CATS] + ["attr_sum"]
    )

    def __init__(self, ontology: Ontology | None = None, seed: int = 13,
                 attributes: pd.DataFrame | None = None):
        if ontology is None:
            self.ont = Ontology(seed, attributes=attributes)
        elif attributes is None:
            self.ont = ontology
        else:
            self.ont = copy.copy(ontology)
            self.ont.attributes = _attributes_index(attributes)
        self.seed = seed

    def _ngrams(self, toks):
        for n in range(self.ont.max_ngram, 0, -1):
            for i in range(len(toks) - n + 1):
                yield " ".join(toks[i:i + n])

    def analyze(self, text: str) -> SymbolicResult:
        toks = TOKEN_RE.findall(text.lower())
        tokset = set(toks)
        f = dict.fromkeys(self.FEATURE_NAMES, 0.0)
        rules: list[Rule] = []

        # 1. Social group mentions
        for ng in set(self._ngrams(toks)):
            info = self.ont.group_terms.get(ng)
            if not info:
                continue
            f[f"grp_{info['category']}"] += 1
            if info["dispreferred"]:
                f["grp_dispreferred"] += 1
                rules.append(Rule("dispreferred_term", ng, f"'{ng}' is a dispreferred descriptor ({info['category']})",
                                  self.ont.source_label(info["source"])))
            else:
                pole = f", {info['pole']}" if info["pole"] else ""
                rules.append(Rule("group_mention", ng, f"social group term ({info['category']}{pole})",
                                  self.ont.source_label(info["source"])))

        # 2. Occupation gender skew and pronoun congruence (WinoBias/Winogender logic)
        fem = bool(tokset & FEMALE_PRON) or any(self.ont.group_terms.get(t, {}).get("pole") == "female" for t in tokset)
        mal = bool(tokset & MALE_PRON) or any(self.ont.group_terms.get(t, {}).get("pole") == "male" for t in tokset)
        for occ, pct in self.ont.occupations.items():
            if occ in tokset or occ + "s" in tokset:
                skew = (pct - 50) / 50  # -1 male-dominated .. +1 female-dominated
                f["occ_present"] = 1
                f["occ_skew"] = max(f["occ_skew"], abs(skew))
                stereo_pole = "female" if skew > 0 else "male"
                if abs(skew) >= 0.4 and (fem or mal) and fem != mal:
                    used = "female" if fem else "male"
                    key = "occ_congruent" if used == stereo_pole else "occ_incongruent"
                    f[key] = 1
                    rules.append(Rule(key, occ, f"'{occ}' is {pct:.0f}% female (BLS); text uses {used} reference",
                                      self.ont.source_label("S3")))

        # 3. Stereotype-bearing attributes (log-odds learned on StereoSet train)
        for c in CATS:
            vals = [(w, self.ont.attributes[(w, c)]) for w in tokset if (w, c) in self.ont.attributes]
            if vals:
                f[f"attr_max_{c}"] = max(v for _, v in vals)
                f[f"attr_min_{c}"] = min(v for _, v in vals)
                f["attr_sum"] += sum(v for _, v in vals)
                for w, v in vals:
                    if abs(v) >= 1.0:
                        kind = "stereotypical" if v > 0 else "anti-stereotypical"
                        rules.append(Rule("stereo_attribute", w, f"'{w}' is {kind} for {c} (log-odds {v:+.2f})",
                                          self.ont.source_label("S4")))
        return SymbolicResult(f, rules)

    def transform(self, texts) -> np.ndarray:
        return np.array([[self.analyze(t).features[k] for k in self.FEATURE_NAMES] for t in texts], dtype=float)

    def transform_crossfit(self, train_df: pd.DataFrame, n_splits: int = 5) -> np.ndarray:
        """S4 attributes of each row are estimated on the other GroupKFold folds."""
        X = np.zeros((len(train_df), len(self.FEATURE_NAMES)), dtype=float)
        if len(train_df) == 0:
            return X
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
                train_df, train_df.label, train_df.group_id):
            attrs = stereo_attributes_from(train_df.iloc[train_idx])
            fold = SymbolicFeaturizer(ontology=self.ont, seed=self.seed, attributes=attrs)
            X[test_idx] = fold.transform(train_df.iloc[test_idx].text)
        return X
