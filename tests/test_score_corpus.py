"""Tests de experiments/score_corpus.py sur des donnees synthetiques (aucun reseau)."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biasdetect.features import neural
from experiments import score_corpus as sc


class FakeSystem:
    name = "Fake"

    def __init__(self):
        self.seed = None
        self.seed_at_fit = None
        self.fit_calls = 0
        self.score_calls = 0

    def fit(self, train, dev):
        self.fit_calls += 1
        self.seed_at_fit = self.seed
        return self

    def score(self, df):
        self.score_calls += 1
        return np.array([len(str(t)) for t in df.text], dtype=float)


class NamedFakeSystem(FakeSystem):
    def __init__(self, name):
        super().__init__()
        self.name = name


def _write_corpus(tmp_path, texts=None):
    texts = texts if texts is not None else ["Hello. World!", "No punctuation", ""]
    corpus = pd.DataFrame({
        "id": [f"item_{i}" for i in range(len(texts))],
        "prompt_id": [f"p{i}" for i in range(len(texts))],
        "text": texts,
    })
    path = tmp_path / "biasdetect_corpus.csv"
    corpus.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest}  biasdetect_corpus.csv\n")
    return corpus, path


def _fake_eval_sets(seed, llm_limit, is_llm):
    train = pd.DataFrame({"text": ["train a", "train b"], "label": [0, 1]})
    dev = pd.DataFrame({"text": ["dev a", "dev b"], "label": [0, 1]})
    return train, dev, {}


def test_split_sentences():
    assert sc.split_sentences("Un. Deux! Trois?") == ["Un.", "Deux!", "Trois?"]
    assert sc.split_sentences("Pas de ponctuation") == ["Pas de ponctuation"]
    assert sc.split_sentences("A.B") == ["A.B"]
    assert sc.split_sentences("Un.   ") == ["Un."]
    assert sc.split_sentences("") == [""]


def test_sentmean_is_mean_of_sentence_scores(monkeypatch, tmp_path):
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")
    fake = FakeSystem()
    texts = ["aa. bbbb!", "ccccc"]

    scores = sc.score_texts(fake, texts, "sentmean")
    assert np.allclose(scores, [4.0, 5.0])
    assert fake.score_calls == 1

    full = sc.score_texts(fake, texts, "full")
    assert np.allclose(full, [9.0, 5.0])
    assert np.allclose(sc.score_texts(fake, [""], "sentmean"), [0.0])


def test_sha256_mismatch_raises_clear_error(tmp_path):
    _, path = _write_corpus(tmp_path)
    (tmp_path / "SHA256SUMS").write_text("0" * 64 + "  biasdetect_corpus.csv\n")

    with pytest.raises(ValueError, match="sha256"):
        sc.verify_sha256sums(tmp_path)

    out = tmp_path / "preds"
    with pytest.raises(ValueError, match="sha256"):
        sc.run([13], path, out)
    assert not out.exists()


def test_run_writes_files_columns_and_refuses_overwrite(monkeypatch, tmp_path):
    corpus, path = _write_corpus(tmp_path)
    out = tmp_path / "preds"
    fake = FakeSystem()
    monkeypatch.setattr(sc, "eval_sets", _fake_eval_sets)
    monkeypatch.setattr(sc.S, "registry", lambda: [fake])
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")

    run_info = sc.run([13], path, out)
    full = out / "Fake__full__s13.csv"
    sentmean = out / "Fake__sentmean__s13.csv"
    assert full.exists() and sentmean.exists()

    scores = pd.read_csv(full)
    assert list(scores.columns) == ["id", "prompt_id", "score", "threshold"]
    assert list(scores["id"]) == list(corpus["id"])
    assert list(scores["prompt_id"]) == list(corpus["prompt_id"])

    on_disk = json.loads(next(out.glob("RUN_INFO_*.json")).read_text())
    assert on_disk == run_info
    assert run_info["seeds"] == [13]
    assert run_info["systems"] == ["Fake"]
    assert run_info["corpus_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert run_info["versions"]["python"]
    assert run_info["versions"]["pandas"]
    assert run_info["versions"]["scikit-learn"]
    assert run_info["date"]
    assert run_info["git_commit"]

    with pytest.raises(FileExistsError, match="overwrite"):
        sc.run([13], path, out)
    sc.run([13], path, out, overwrite=True)


def test_seed_is_set_before_fit(monkeypatch, tmp_path):
    _, path = _write_corpus(tmp_path)
    fake = FakeSystem()
    monkeypatch.setattr(sc, "eval_sets", _fake_eval_sets)
    monkeypatch.setattr(sc.S, "registry", lambda: [fake])
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")

    sc.run([42], path, tmp_path / "preds")

    assert fake.fit_calls == 1
    assert fake.seed_at_fit == 42


def test_only_filters_registry(monkeypatch, tmp_path):
    _, path = _write_corpus(tmp_path)
    out = tmp_path / "preds"
    kept, dropped = NamedFakeSystem("A"), NamedFakeSystem("B")
    monkeypatch.setattr(sc, "eval_sets", _fake_eval_sets)
    monkeypatch.setattr(sc.S, "registry", lambda: [kept, dropped])
    monkeypatch.setattr(neural, "CACHE", tmp_path / "cache")

    run_info = sc.run([13], path, out, only=["A"])

    assert run_info["systems"] == ["A"]
    assert (out / "A__full__s13.csv").exists()
    assert not (out / "B__full__s13.csv").exists()
    assert dropped.fit_calls == 0
