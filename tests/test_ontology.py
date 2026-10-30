"""Tests de l'ontologie RDF et du featurizer symbolique (mini-ontologie dans tmp_path)."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from rdflib import Graph, Literal
from rdflib.namespace import PROV, RDF, RDFS

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.features import symbolic
from biasdetect.ontology import build


def _write_mini_ontology(path):
    g = Graph()
    g.bind("bd", build.BD)
    g.bind("prov", PROV)

    def term(label, category, pole=None, preference=None, source="S1"):
        u = build._uri("term", label)
        g.add((u, RDF.type, build.BD.SocialGroupTerm))
        g.add((u, RDFS.label, Literal(label.lower())))
        g.add((u, build.BD.category, Literal(category)))
        if pole:
            g.add((u, build.BD.pole, Literal(pole)))
        if preference:
            g.add((u, build.BD.preference, Literal(preference)))
        g.add((u, PROV.wasDerivedFrom, build.BD[source]))

    def occupation(label, pct_female):
        u = build._uri("occ", label)
        g.add((u, RDF.type, build.BD.Occupation))
        g.add((u, RDFS.label, Literal(label.lower())))
        g.add((u, build.BD.pctFemale, Literal(float(pct_female))))
        g.add((u, PROV.wasDerivedFrom, build.BD.S3))

    def attribute(word, category, log_odds):
        u = build._uri(f"attr_{category}", word)
        g.add((u, RDF.type, build.BD.StereoAttribute))
        g.add((u, RDFS.label, Literal(word.lower())))
        g.add((u, build.BD.category, Literal(category)))
        g.add((u, build.BD.logOdds, Literal(float(log_odds))))
        g.add((u, PROV.wasDerivedFrom, build.BD.S4))

    term("woman", "gender", pole="female", source="S1")
    term("man", "gender", pole="male", source="S1")
    term("illegal alien", "origin", preference="dispreferred", source="S2")
    occupation("nurse", 90.0)
    occupation("doctor", 30.0)
    attribute("emotional", "gender", 1.5)
    attribute("logical", "gender", -1.2)
    g.serialize(path, format="turtle")


@pytest.fixture
def mini_ttl(tmp_path, monkeypatch):
    path = tmp_path / "mini.ttl"
    _write_mini_ontology(path)

    def fail_build(*args, **kwargs):
        raise AssertionError("build() ne doit pas etre appele : la mini-ontologie existe")

    monkeypatch.setattr(symbolic, "ttl_path", lambda seed=13: path)
    monkeypatch.setattr(symbolic, "build", fail_build)
    return path


@pytest.fixture
def featurizer(mini_ttl):
    return symbolic.SymbolicFeaturizer(seed=13)


# ---------- ontology/build.py ----------

def test_uri_normalizes_terms():
    assert build._uri("term", "Illegal Alien!") == build.BD["term_illegal_alien_"]
    assert build._uri("occ", "Registered Nurse") == build.BD["occ_registered_nurse"]
    assert build._uri("term", "  Woman ") == build.BD["term_woman"]


def test_hb_descriptors_flattens_nested_structures():
    data = {
        "axis": [
            {"descriptor": "Black", "preference": "reviewed"},
            {"descriptor": "colored", "preference": "dispreferred"},
        ],
        "other": ["Hispanic", "Latino"],
    }
    assert list(build._hb_descriptors(data)) == [
        ("Black", "reviewed"),
        ("colored", "dispreferred"),
        ("Hispanic", "reviewed"),
        ("Latino", "reviewed"),
    ]


def test_stereo_attributes_log_odds_and_train_filter(monkeypatch):
    df = pd.DataFrame({
        "text": ["happy family"] * 3 + ["sad person"] * 2 + ["zzz test"],
        "label": [1, 1, 1, 0, 0, 1],
        "split": ["train"] * 5 + ["test"],
        "category": ["gender"] * 6,
    })
    calls = []

    def fake_load(name, seed=13):
        calls.append((name, seed))
        return df

    monkeypatch.setattr(build, "load_processed", fake_load)

    out = build.stereo_attributes(seed=13, min_count=3)
    assert calls == [("stereoset", 13)]
    assert set(out["word"]) == {"happy", "family"}
    happy = out[out.word == "happy"].iloc[0]
    assert happy["log_odds"] == pytest.approx(math.log(12))
    assert happy["n_stereo"] == 3
    assert happy["n_anti"] == 0
    assert happy["category"] == "gender"

    out = build.stereo_attributes(seed=13, min_count=2)
    sad = out[out.word == "sad"].iloc[0]
    assert sad["log_odds"] == pytest.approx(-math.log(12))
    assert "zzz" not in set(out["word"])


# ---------- features/symbolic.py ----------

def test_ontology_indexes_terms_occupations_and_attributes(featurizer):
    ont = featurizer.ont
    assert ont.max_ngram == 2
    assert ont.group_terms["woman"] == {
        "category": "gender",
        "pole": "female",
        "dispreferred": False,
        "source": "S1",
    }
    assert ont.group_terms["illegal alien"]["dispreferred"] is True
    assert ont.group_terms["illegal alien"]["category"] == "origin"
    assert ont.occupations["nurse"] == pytest.approx(90.0)
    assert ont.attributes[("emotional", "gender")] == pytest.approx(1.5)
    assert ont.source_label("S1") == build.SOURCES["S1"][0]


def test_analyze_group_occupation_and_attribute_rules(featurizer):
    res = featurizer.analyze("The woman is a nurse and she is emotional.")
    f = res.features
    assert f["grp_gender"] == 1
    assert f["occ_present"] == 1
    assert f["occ_skew"] == pytest.approx(0.8)
    assert f["occ_congruent"] == 1
    assert f["occ_incongruent"] == 0
    assert f["attr_max_gender"] == pytest.approx(1.5)
    assert f["attr_min_gender"] == pytest.approx(1.5)
    assert f["attr_sum"] == pytest.approx(1.5)
    kinds = [r.rule for r in res.rules]
    assert "group_mention" in kinds
    assert "occ_congruent" in kinds
    assert "stereo_attribute" in kinds


def test_analyze_dispreferred_multiword_term(featurizer):
    res = featurizer.analyze("An illegal alien arrived.")
    assert res.features["grp_origin"] == 1
    assert res.features["grp_dispreferred"] == 1
    rules = [r for r in res.rules if r.rule == "dispreferred_term"]
    assert len(rules) == 1
    assert rules[0].term == "illegal alien"
    assert rules[0].source == build.SOURCES["S2"][0]


def test_transform_matches_analyze(featurizer):
    texts = ["The woman is a nurse.", "An illegal alien arrived."]
    X = featurizer.transform(texts)
    assert X.shape == (2, len(symbolic.SymbolicFeaturizer.FEATURE_NAMES))
    expected = [
        [featurizer.analyze(t).features[k] for k in symbolic.SymbolicFeaturizer.FEATURE_NAMES]
        for t in texts
    ]
    assert np.allclose(X, np.array(expected, dtype=float))
