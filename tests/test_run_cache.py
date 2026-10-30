"""Tests du cache de predictions de experiments/run.py (aucun modele, aucun reseau)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import run


class FakeSystem:
    name = "Fake"

    def __init__(self):
        self.calls = 0
        self.version = 0

    def fit(self, train, dev):
        self.version += 1
        return self

    def score(self, df):
        self.calls += 1
        return np.linspace(0.1, 0.9, len(df), dtype=float) * self.version


def _frame(n):
    return pd.DataFrame({
        "text": [f"text-{i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
        "group_id": [f"g{i}" for i in range(n)],
    })


def _install_eval(monkeypatch, train, dev, sets):
    monkeypatch.setattr(run, "eval_sets", lambda seed, llm_limit, is_llm: (train, dev, sets))


def _cache_path(tmp_path):
    return tmp_path / "preds" / "Fake__stereoset_test__s13.csv"


def test_cache_recomputed_when_length_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "PRED", tmp_path / "preds")
    train, dev = _frame(2), _frame(2)
    fake = FakeSystem()

    _install_eval(monkeypatch, train, dev, {"stereoset_test": _frame(4)})
    run.run_system(fake, 13)
    path = _cache_path(tmp_path)
    assert len(pd.read_csv(path)) == 4

    calls_before = fake.calls
    _install_eval(monkeypatch, train, dev, {"stereoset_test": _frame(3)})
    rows = run.run_system(fake, 13)

    assert fake.calls == calls_before + 2
    assert len(pd.read_csv(path)) == 3
    assert rows[0]["n"] == 3


def test_no_cache_recomputes_existing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "PRED", tmp_path / "preds")
    train, dev = _frame(2), _frame(2)
    fake = FakeSystem()

    _install_eval(monkeypatch, train, dev, {"stereoset_test": _frame(4)})
    run.run_system(fake, 13)
    path = _cache_path(tmp_path)
    first = pd.read_csv(path).score.iloc[0]

    calls_before = fake.calls
    run.run_system(fake, 13, no_cache=True)

    assert fake.calls == calls_before + 2
    second = pd.read_csv(path).score.iloc[0]
    assert second != pytest.approx(first)


def test_cached_threshold_is_reused(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "PRED", tmp_path / "preds")
    train, dev = _frame(2), _frame(2)
    fake = FakeSystem()

    _install_eval(monkeypatch, train, dev, {"stereoset_test": _frame(4)})
    run.run_system(fake, 13)
    path = _cache_path(tmp_path)
    cached = pd.read_csv(path)
    cached["threshold"] = 0.123
    cached.to_csv(path, index=False)

    rows = run.run_system(fake, 13)

    assert rows[0]["threshold"] == pytest.approx(0.123)
