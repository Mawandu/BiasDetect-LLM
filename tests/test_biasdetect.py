"""Tests unitaires de `biasdetect/` (aucun appel reseau, aucun chargement de modele)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.annotation import agreement
from biasdetect.data.loaders import _sbic_category, group_split
from biasdetect.eval import metrics as eval_metrics
from biasdetect.models import systems


class DummyFeaturizer:
    """Faux `SymbolicFeaturizer` : memorise la seed recue et renvoie des features factices."""

    FEATURE_NAMES = ("dummy_a", "dummy_b")
    calls = []

    def __init__(self, ontology=None, seed=13):
        self.seed = seed
        DummyFeaturizer.calls.append(seed)

    def transform(self, texts):
        texts = list(texts)
        return np.column_stack([np.arange(len(texts), dtype=float), np.ones(len(texts))])

    def transform_crossfit(self, train_df, n_splits=5):
        return self.transform(train_df.text)

    def analyze(self, text):
        raise NotImplementedError


@pytest.fixture
def toy_split():
    train = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    dev = pd.DataFrame({"text": ["e", "f"], "label": [0, 1]})
    return train, dev


# ---------- SymbolicLR : propagation de la seed (pas de fuite de donnees) ----------

def test_symbolic_lr_passes_seed_to_featurizer(monkeypatch, toy_split):
    """`SymbolicLR.fit` doit transmettre sa seed a `SymbolicFeaturizer` (evite la fuite S4)."""
    train, dev = toy_split
    DummyFeaturizer.calls = []
    monkeypatch.setattr(systems, "SymbolicFeaturizer", DummyFeaturizer)

    sysobj = systems.SymbolicLR()
    sysobj.seed = 21
    sysobj.fit(train, dev)

    assert DummyFeaturizer.calls == [21]


def test_symbolic_lr_default_seed_is_13(monkeypatch, toy_split):
    """Sans seed explicite, `SymbolicLR` utilise la seed 13 (ontologie `bias_s13.ttl`)."""
    train, dev = toy_split
    DummyFeaturizer.calls = []
    monkeypatch.setattr(systems, "SymbolicFeaturizer", DummyFeaturizer)

    sysobj = systems.SymbolicLR()
    sysobj.fit(train, dev)

    assert DummyFeaturizer.calls == [13]


# ---------- agreement : ecriture du gold sans toucher a data/ ----------

def test_agreement_resolve_creates_parent_and_writes_gold_in_tmp_path(monkeypatch, tmp_path):
    """`--resolve` cree `data/bold/` et ecrit `gold.csv` (redirige vers `tmp_path`)."""
    ann = tmp_path / "annotation"
    ann.mkdir()
    items = pd.DataFrame({
        "item_id": ["it0000", "it0001"],
        "text": ["a", "b"],
        "label": ["0", "1"],
        "category": ["none", "gender"],
        "domain": ["gender", "race"],
    })
    items[["item_id", "text", "label", "category"]].to_csv(ann / "to_annotate_A1.csv", index=False)
    items[["item_id", "text", "label", "category"]].to_csv(ann / "to_annotate_A2.csv", index=False)
    items[["item_id", "domain"]].to_csv(ann / "items_master.csv", index=False)
    pd.DataFrame(columns=["item_id", "gold"]).to_csv(ann / "disagreements.csv", index=False)

    monkeypatch.setattr(agreement, "ROOT", tmp_path)
    monkeypatch.setattr(agreement, "ANN", ann)
    monkeypatch.setattr(sys, "argv", ["agreement", "--resolve"])

    agreement.main()

    out = tmp_path / "data" / "bold" / "gold.csv"
    assert out.parent.is_dir()
    assert out.exists()
    gold = pd.read_csv(out)
    assert len(gold) == 2
    assert list(gold["label"]) == [0, 1]


# ---------- eval/metrics.py ----------

def test_best_threshold_separates_perfectly():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    thr = eval_metrics.best_threshold(y, s)
    assert 0.2 < thr <= 0.8
    assert (s >= thr).tolist() == [False, False, True, True]


def test_ece_calibrated_and_miscalibrated():
    assert eval_metrics.ece(np.array([0, 1]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert eval_metrics.ece(np.array([0, 0, 1, 1]), np.array([0.9, 0.9, 0.9, 0.9])) == pytest.approx(0.4)


def test_ece_weights_bins_by_frequency():
    y = np.array([0, 0, 0, 1])
    p = np.array([0.15, 0.15, 0.15, 0.85])
    assert eval_metrics.ece(y, p) == pytest.approx(0.15)


def test_metrics_perfect_scores():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    out = eval_metrics.metrics(y, s, 0.5)
    assert out["n"] == 4
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["auprc"] == pytest.approx(1.0)
    assert 0.0 <= out["ece"] <= 1.0


def test_metrics_handles_nan_and_single_class():
    out = eval_metrics.metrics(np.array([0, 1]), np.array([np.nan, 0.7]), 0.5)
    assert out["n"] == 1
    assert np.isnan(out["auroc"])
    assert np.isnan(out["auprc"])

    out = eval_metrics.metrics(np.array([0, 1]), np.array([-0.5, 1.5]), 0.5)
    assert np.isnan(out["ece"])


def test_pairwise_accuracy_counts_ties_as_half():
    df = pd.DataFrame({
        "group_id": ["a", "a", "b", "b", "c", "c"],
        "label": [0, 1, 0, 1, 0, 1],
    })
    s = np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.1])
    assert eval_metrics.pairwise_accuracy(df, s) == pytest.approx(0.5)


def test_bootstrap_ci_deterministic_on_perfect_scores():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.2, 0.8])
    lo, hi = eval_metrics.bootstrap_ci(y, s, 0.5, n=10, seed=0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)
    assert eval_metrics.bootstrap_ci(y, s, 0.5, n=10, seed=0) == pytest.approx((lo, hi))


def test_mcnemar_identical_and_discordant():
    y = np.zeros(4, dtype=int)
    same = np.zeros(4, dtype=int)
    wrong = np.ones(4, dtype=int)
    assert eval_metrics.mcnemar(y, same, same) == 1.0
    assert eval_metrics.mcnemar(y, same, wrong) == pytest.approx(0.125)


# ---------- data/loaders.py (sans reseau) ----------

@pytest.mark.parametrize(("cats", "expected"), [
    (["gender"], "gender"),
    (["race", "gender"], "gender"),
    (["race"], "origin"),
    (["culture"], "origin"),
    (["religion"], "religion"),
    (["age", "disability"], None),
    ([], None),
])
def test_sbic_category(cats, expected):
    assert _sbic_category(cats) == expected


def _synthetic_groups():
    rows = []
    for cat in ("gender", "origin", "religion"):
        for g in range(10):
            gid = f"{cat}_g{g}"
            for k in range(2):
                rows.append({"text": f"{gid}_{k}", "label": k, "category": cat, "group_id": gid})
    return pd.DataFrame(rows)


def test_group_split_has_no_group_id_leakage():
    df = _synthetic_groups()
    out = group_split(df, seed=13)
    assert set(out["split"]) == {"train", "dev", "test"}
    by_split = {
        name: set(out.loc[out.split == name, "group_id"])
        for name in ("train", "dev", "test")
    }
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["dev"].isdisjoint(by_split["test"])
    assert by_split["train"] | by_split["dev"] | by_split["test"] == set(df["group_id"])


def test_group_split_is_deterministic_and_stratified():
    df = _synthetic_groups()
    a = group_split(df, seed=13)
    b = group_split(df, seed=13)
    pd.testing.assert_series_equal(a["split"], b["split"])
    for _, sub in a.groupby("category"):
        assert set(sub["split"]) == {"train", "dev", "test"}
