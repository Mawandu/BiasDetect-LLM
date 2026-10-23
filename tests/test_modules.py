import os
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import linguistic
from modules.fusion import FusionEngine


# ---------- Fusion ----------

def test_fusion_weighted_average():
    res = FusionEngine().fuse_scores({"score": 60}, {"score": 20, "issues": []}, alpha=0.5)
    assert res["final_score"] == 40


def test_fusion_no_bias_message():
    res = FusionEngine().fuse_scores({"score": 0}, {"score": 0, "issues": []})
    assert "Aucun biais" in res["explanation"]


# ---------- Symbolic ----------

@pytest.fixture(scope="module")
def sym():
    from modules.symbolic import SymbolicAnalyzer
    sym_analyzer = SymbolicAnalyzer()
    if sym_analyzer.nlp is None:
        pytest.skip("en_core_web_sm not installed")
    return sym_analyzer


def concepts(res):
    return [i["concept"] for i in res["issues"]]


def test_symbolic_single_word(sym):
    assert "doctor" in concepts(sym.analyze_text("The doctor said he was busy."))


def test_symbolic_multi_word(sym):
    res = sym.analyze_text("The flight attendant talked about an illegal alien.")
    assert "flight attendant" in concepts(res)
    assert "illegal alien" in concepts(res)
    # 'critical' severity -> +20, profession -> +10
    assert res["score"] == 30


def test_symbolic_plural(sym):
    assert "nurse" in concepts(sym.analyze_text("Nurses work hard."))


def test_symbolic_empty(sym):
    assert sym.analyze_text("   ")["score"] == 0


class FakeSentenceTransformer:
    MALE_WORDS = {"he", "him", "his", "man", "boy", "father", "son", "male"}
    FEMALE_WORDS = {"she", "her", "woman", "girl", "mother", "daughter", "female"}

    def __init__(self, model_name):
        self.model_name = model_name
        self.encode_calls = 0

    def encode(self, inputs, **kwargs):
        self.encode_calls += 1
        if isinstance(inputs, str):
            return self._vector_for_text(inputs)
        return np.array([self._vector_for_text(t) for t in inputs], dtype=float)

    @classmethod
    def _vector_for_text(cls, text):
        words = set(re.findall(r"[a-z']+", text.lower()))
        male = bool(words & cls.MALE_WORDS)
        female = bool(words & cls.FEMALE_WORDS)
        if male and not female:
            return [1.0, 0.0]
        if female and not male:
            return [0.0, 1.0]
        return [1.0, 1.0]


@pytest.fixture
def fake_encoder(monkeypatch):
    monkeypatch.setattr(linguistic, "SentenceTransformer", FakeSentenceTransformer)
    return FakeSentenceTransformer


def test_linguistic_uses_mocked_model(fake_encoder):
    analyzer = linguistic.LinguisticAnalyzer()
    assert isinstance(analyzer.model, FakeSentenceTransformer)
    assert analyzer.model.model_name == "all-MiniLM-L6-v2"


def test_linguistic_empty_text(fake_encoder):
    res = linguistic.LinguisticAnalyzer().analyze_text("   ")
    assert res["score"] == 0.0
    assert res["direction"] == "Neutre"
    assert res["raw_diff"] == 0.0
    assert res["details"] == "Empty text"


def test_linguistic_male_bias(fake_encoder):
    res = linguistic.LinguisticAnalyzer().analyze_text("The man is a father.")
    assert res["direction"] == "Masculin"
    assert res["raw_diff"] == 1.0
    assert res["score"] == 100.0


def test_linguistic_female_bias(fake_encoder):
    res = linguistic.LinguisticAnalyzer().analyze_text("The woman is a mother.")
    assert res["direction"] == "Féminin"
    assert res["raw_diff"] == -1.0
    assert res["score"] == 100.0


def test_linguistic_neutral_text(fake_encoder):
    res = linguistic.LinguisticAnalyzer().analyze_text("The weather is nice today.")
    assert res["direction"] == "Neutre"
    assert res["raw_diff"] == 0.0
    assert res["score"] == 0.0
