"""Tests des features neuronales et des modeles (encodeur et featurizer factices)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.features import neural
from biasdetect.features.symbolic import Rule, SymbolicResult
from biasdetect.models import systems


class FakeEncoder:
    def __init__(self, dim=3):
        self.dim = dim
        self.calls = []

    def encode(self, texts, **kwargs):
        texts = list(texts)
        self.calls.append(texts)
        return np.array(
            [[float(len(t)), float(t.count("a")), float(i + 1)] for i, t in enumerate(texts)],
            dtype=float,
        )


def _fake_embed(texts, key, **kwargs):
    out = []
    for text in texts:
        text = str(text)
        if text.startswith("pos"):
            out.append([2.0, -0.1])
        elif text.startswith("neg"):
            out.append([-2.0, 0.1])
        else:
            out.append([0.0, 0.0])
    return np.array(out, dtype=float)


def _neutral_embed(texts, key, **kwargs):
    return np.zeros((len(list(texts)), 2), dtype=float)


_FEATURES = {
    "a": [0.0, 0.1],
    "b": [0.2, 0.1],
    "c": [3.0, 0.1],
    "d": [3.2, 0.1],
    "e": [3.1, 0.1],
    "f": [-0.1, 0.1],
}


class DummyFeaturizer:
    FEATURE_NAMES = ("dummy_a", "dummy_b")

    def __init__(self, ontology=None, seed=13):
        self.seed = seed

    def transform(self, texts):
        out = []
        for text in texts:
            text = str(text)
            if text.startswith("pos"):
                out.append([1.0, 0.0])
            elif text.startswith("neg"):
                out.append([0.0, 0.0])
            else:
                out.append(_FEATURES.get(text, [0.0, 0.0]))
        return np.array(out, dtype=float)

    def transform_crossfit(self, train_df, n_splits=5):
        return self.transform(train_df.text)

    def analyze(self, text):
        return SymbolicResult({}, [Rule("dummy", str(text), "fake rule", "S1")])


# ---------- features/neural.py ----------

def test_embed_uses_encoder_and_writes_cache_in_tmp_path(tmp_path, monkeypatch):
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")
    encoder = FakeEncoder()
    monkeypatch.setattr(neural, "_get_model", lambda key: encoder)

    e1 = neural.embed(["alpha", "ab"], "minilm")
    assert e1.shape == (2, 3)
    assert len(encoder.calls) == 1
    assert len(list((tmp_path / "cache").glob("minilm_*.npy"))) == 1

    e2 = neural.embed(["alpha", "ab"], "minilm")
    assert np.allclose(e1, e2)
    assert len(encoder.calls) == 1


def test_embed_applies_e5_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")
    encoder = FakeEncoder()
    monkeypatch.setattr(neural, "_get_model", lambda key: encoder)

    neural.embed(["hello"], "e5")
    assert encoder.calls == [["query: hello"]]


# ---------- models/systems.py ----------

def test_majority_returns_train_prevalence():
    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    dev = pd.DataFrame({"text": ["e"], "label": [0]})
    sys_obj = systems.Majority().fit(train, dev)
    assert np.allclose(sys_obj.score(dev), [0.5])


def test_neural_probe_fit_score(monkeypatch):
    monkeypatch.setattr(neural, "embed", _fake_embed)
    train = pd.DataFrame({
        "text": ["neg0", "neg1", "neg2", "neg3", "pos0", "pos1", "pos2", "pos3"],
        "label": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    dev = pd.DataFrame({"text": ["neg9", "pos9"], "label": [0, 1]})
    sys_obj = systems.NeuralProbe(enc="minilm").fit(train, dev)
    scores = sys_obj.score(dev)
    assert scores[0] < 0.5 < scores[1]


def test_symbolic_lr_fit_score(monkeypatch):
    monkeypatch.setattr(systems, "SymbolicFeaturizer", DummyFeaturizer)
    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 0, 1, 1]})
    dev = pd.DataFrame({"text": ["f", "e"], "label": [0, 1]})
    sys_obj = systems.SymbolicLR().fit(train, dev)
    scores = sys_obj.score(dev)
    assert scores[0] < 0.5 < scores[1]


def test_hybrid_fusion_fit_score_explain(monkeypatch):
    monkeypatch.setattr(neural, "embed", _fake_embed)
    monkeypatch.setattr(systems, "SymbolicFeaturizer", DummyFeaturizer)

    rows = [
        {"text": ("pos" if i % 2 else "neg") + str(i), "label": i % 2, "group_id": f"g{i}"}
        for i in range(10)
    ]
    train = pd.DataFrame(rows)
    dev = pd.DataFrame({"text": ["neg9", "pos9"], "label": [0, 1]})

    sys_obj = systems.HybridFusion(encs=["minilm"]).fit(train, dev)
    scores = sys_obj.score(dev)
    assert scores[0] < 0.5 < scores[1]

    explained = sys_obj.explain("pos9")
    assert isinstance(explained["score"], float)
    assert explained["score"] > 0.5
    assert all(isinstance(v, float) for v in explained["contributions"].values())
    assert [r["rule"] for r in explained["rules"]] == ["dummy"]


def test_majority_returns_non_trivial_prevalence():
    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [1, 1, 1, 0]})
    dev = pd.DataFrame({"text": ["e"], "label": [0]})
    sys_obj = systems.Majority().fit(train, dev)
    assert np.allclose(sys_obj.score(dev), [0.75])


def test_hybrid_fusion_relies_on_symbolic_features(monkeypatch):
    monkeypatch.setattr(neural, "embed", _neutral_embed)
    monkeypatch.setattr(systems, "SymbolicFeaturizer", DummyFeaturizer)

    train = pd.DataFrame([
        {"text": ("pos" if i % 2 else "neg") + str(i), "label": i % 2, "group_id": f"g{i}"}
        for i in range(10)
    ])
    dev = pd.DataFrame({"text": ["neg20", "pos20"], "label": [0, 1]})

    sys_obj = systems.HybridFusion(encs=["minilm"]).fit(train, dev)
    scores = sys_obj.score(dev)
    assert scores[0] < 0.5 < scores[1]

    weights = sys_obj.branch_weights()
    assert abs(weights["dummy_a"]) > 0.1
