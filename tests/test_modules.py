import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("en_core_web_sm not installed")
    from modules.symbolic import SymbolicAnalyzer
    return SymbolicAnalyzer()


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
