"""Tests de la fuite S4 (attributs calcules sur leurs propres etiquettes) et du cross-fitting."""
import hashlib
import os
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.features import neural, symbolic
from biasdetect.models import systems
from biasdetect.ontology import build

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "stereoset_s13.csv"

FROZEN_SHAPE = (740, 5)
FROZEN_SHA256 = "3d89758079c725af59b68ee45e86627aea623fc69c8af42608e3b36097e6dfa0"


class DummyOntology:
    max_ngram = 1
    group_terms = {}
    occupations = {}
    attributes = {}

    def source_label(self, sid):
        return sid


def _noise_frame(seed=0, n=200, vocab=100, words_per_text=12):
    """Etiquettes en bruit pur, mots aleatoires : toute correlation est une fuite."""
    rng = np.random.default_rng(seed)
    words = [a + b for a in string.ascii_lowercase for b in string.ascii_lowercase][:vocab]
    rows = []
    for i in range(n):
        toks = rng.choice(words, size=words_per_text, replace=False)
        rows.append({
            "text": " ".join(toks),
            "label": int(rng.integers(0, 2)),
            "category": "gender",
            "split": "train",
            "group_id": f"g{i}",
        })
    return pd.DataFrame(rows)


def _constant_embed(texts, key, **kwargs):
    return np.zeros((len(list(texts)), 2), dtype=float)


class SpyFeaturizer:
    FEATURE_NAMES = symbolic.SymbolicFeaturizer.FEATURE_NAMES

    def __init__(self, inner):
        self.inner = inner
        self.crossfit_calls = 0
        self.transform_calls = 0

    def transform(self, texts):
        self.transform_calls += 1
        return self.inner.transform(texts)

    def transform_crossfit(self, train_df, n_splits=5):
        self.crossfit_calls += 1
        return self.inner.transform_crossfit(train_df, n_splits=n_splits)


def _in_sample_attr_sum(df):
    """attr_sum calcule avec les attributs estimes sur les memes lignes (fuite)."""
    attrs = build.stereo_attributes(13)
    index = {(r.word, r.category): float(r.log_odds) for r in attrs.itertuples()
             if abs(float(r.log_odds)) >= 0.5}
    out = []
    for text in df.text:
        toks = set(symbolic.TOKEN_RE.findall(text.lower()))
        out.append(sum(index.get((w, "gender"), 0.0) for w in toks))
    return np.array(out, dtype=float)


@pytest.mark.skipif(not PROCESSED.exists(), reason="data/processed/stereoset_s13.csv absent (pas de telechargement)")
def test_stereo_attributes_regression_frozen_values():
    out = build.stereo_attributes(13)
    assert out.shape == FROZEN_SHAPE
    canon = out.sort_values(["category", "word"])
    payload = "\n".join(
        f"{r.category}|{r.word}|{float(r.log_odds):.12f}|{int(r.n_stereo)}|{int(r.n_anti)}"
        for r in canon.itertuples()
    )
    assert hashlib.sha256(payload.encode()).hexdigest() == FROZEN_SHA256


def test_s4_leak_in_sample_then_removed_by_crossfit(monkeypatch):
    df = _noise_frame()
    monkeypatch.setattr(build, "load_processed", lambda *args, **kwargs: df)

    auc_in = roc_auc_score(df.label, _in_sample_attr_sum(df))
    print(f"AUROC in-sample={auc_in:.3f}")
    assert auc_in > 0.65, f"la fuite en echantillon devrait etre forte (AUROC={auc_in:.3f})"

    full = build.stereo_attributes_from(df)
    fz = symbolic.SymbolicFeaturizer(ontology=DummyOntology(), attributes=full)
    X = fz.transform_crossfit(df)
    idx = symbolic.SymbolicFeaturizer.FEATURE_NAMES.index("attr_sum")
    auc_cf = roc_auc_score(df.label, X[:, idx])
    print(f"AUROC crossfit={auc_cf:.3f}")
    assert 0.45 <= auc_cf <= 0.55, f"le cross-fitting doit supprimer la fuite (AUROC={auc_cf:.3f})"

    captured = {}

    class Capture:
        def fit(self, X, y):
            captured["X"] = np.asarray(X)
            return self

    monkeypatch.setattr(systems, "SymbolicFeaturizer", lambda seed=13: fz)
    monkeypatch.setattr(systems, "make_pipeline", lambda *args, **kwargs: Capture())
    systems.SymbolicLR().fit(df, df)
    auc_fit = roc_auc_score(df.label, captured["X"][:, idx])
    print(f"AUROC SymbolicLR.fit={auc_fit:.3f}")
    assert 0.45 <= auc_fit <= 0.55, f"SymbolicLR.fit doit utiliser le cross-fitting (AUROC={auc_fit:.3f})"


def test_hybrid_fusion_s4_training_scores_no_leak(monkeypatch):
    df = _noise_frame(n=1000, vocab=400, words_per_text=12)
    full = build.stereo_attributes_from(df)
    fz = symbolic.SymbolicFeaturizer(ontology=DummyOntology(), attributes=full)

    class Factory:
        FEATURE_NAMES = symbolic.SymbolicFeaturizer.FEATURE_NAMES

        def __new__(cls, *args, **kwargs):
            return fz

    real_make_pipeline = systems.make_pipeline
    captured = {}

    def capturing_make_pipeline(*args, **kwargs):
        pipe = real_make_pipeline(*args, **kwargs)
        original_fit = pipe.fit

        def fit(X, y, **kw):
            captured["X"] = np.asarray(X)
            return original_fit(X, y, **kw)

        pipe.fit = fit
        return pipe

    monkeypatch.setattr(neural, "embed", _constant_embed)
    monkeypatch.setattr(systems, "SymbolicFeaturizer", Factory)
    monkeypatch.setattr(systems, "make_pipeline", capturing_make_pipeline)

    model = systems.HybridFusion(encs=["minilm"]).fit(df, df)
    X = captured["X"]
    Z = model.clf[0].transform(X)
    coef = model.clf[-1].coef_[0]
    attr_idx = [i for i, name in enumerate(model.feature_names) if name.startswith("attr_")]
    auc = roc_auc_score(df.label, Z[:, attr_idx] @ coef[attr_idx])
    print(f"AUROC HybridFusion S4={auc:.3f}")
    assert 0.45 <= auc <= 0.55, f"HybridFusion ne doit pas exploiter la fuite S4 (AUROC={auc:.3f})"


def test_hybrid_fusion_fit_calls_transform_crossfit(monkeypatch):
    df = _noise_frame(n=40, vocab=20, words_per_text=6)
    full = build.stereo_attributes_from(df)
    spy = SpyFeaturizer(symbolic.SymbolicFeaturizer(ontology=DummyOntology(), attributes=full))

    class Factory:
        FEATURE_NAMES = symbolic.SymbolicFeaturizer.FEATURE_NAMES

        def __new__(cls, *args, **kwargs):
            return spy

    monkeypatch.setattr(neural, "embed", _constant_embed)
    monkeypatch.setattr(systems, "SymbolicFeaturizer", Factory)

    systems.HybridFusion(encs=["minilm"]).fit(df, df)

    assert spy.crossfit_calls == 1
    assert spy.transform_calls == 0


def test_transform_crossfit_shape_finite_and_disjoint_groups(monkeypatch):
    df = _noise_frame(n=60, vocab=20, words_per_text=6)
    full = build.stereo_attributes_from(df)
    fz = symbolic.SymbolicFeaturizer(ontology=DummyOntology(), attributes=full)
    X_full = fz.transform(df.text)

    seen = []
    original = symbolic.stereo_attributes_from

    def spy(sub, *args, **kwargs):
        seen.append(set(sub.group_id))
        return original(sub, *args, **kwargs)

    monkeypatch.setattr(symbolic, "stereo_attributes_from", spy)
    X_cf = fz.transform_crossfit(df, n_splits=5)

    assert X_cf.shape == X_full.shape
    assert np.isfinite(X_cf).all()

    names = symbolic.SymbolicFeaturizer.FEATURE_NAMES
    non_attr = [i for i, name in enumerate(names) if not name.startswith("attr_")]
    assert np.allclose(X_cf[:, non_attr], X_full[:, non_attr])

    folds = list(GroupKFold(5).split(df, df.label, df.group_id))
    assert len(seen) == 5
    for (_, test_idx), estimation_groups in zip(folds, seen):
        assert estimation_groups.isdisjoint(set(df.iloc[test_idx].group_id))
