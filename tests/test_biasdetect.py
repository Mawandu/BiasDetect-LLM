"""Tests unitaires de `biasdetect/` (aucun appel reseau, aucun chargement de modele)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
