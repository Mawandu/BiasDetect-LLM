"""Tests unitaires de `biasdetect/` (aucun appel reseau, aucun chargement de modele)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.annotation import agreement
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
