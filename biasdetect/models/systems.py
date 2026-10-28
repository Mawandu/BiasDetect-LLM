"""All systems compared in the paper.

Each system implements fit(train_df, dev_df) and score(df) -> np.ndarray of
continuous scores where higher = more stereotypical. Thresholds are chosen on dev.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from biasdetect.features import neural
from biasdetect.features.symbolic import SymbolicFeaturizer

ROOT = Path(__file__).resolve().parents[2]


class System:
    name = "system"
    seed = 13  # split seed, set by the runner (ontology S4 is rebuilt per split)

    def fit(self, train: pd.DataFrame, dev: pd.DataFrame):
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


# ------------------------------------------------------------------ B0
class Majority(System):
    name = "B0_majority"

    def fit(self, train, dev):
        self.p = train.label.mean()
        return self

    def score(self, df):
        return np.full(len(df), self.p)


# ------------------------------------------------------------------ B1 / B2: the original demo
class LegacyRules(System):
    """B1: the original hand-written dictionary of the demo (modules/symbolic.py)."""
    name = "B1_legacy_rules"

    def fit(self, train, dev):
        sys.path.insert(0, str(ROOT))
        from modules.symbolic import SymbolicAnalyzer
        self.a = SymbolicAnalyzer()
        return self

    def score(self, df):
        return np.array([self.a.analyze_text(t)["score"] for t in df.text], dtype=float)


class LegacyGenderSim(System):
    """B2: the original neural branch (whole-sentence cosine to he/she word sets)."""
    name = "B2_legacy_embsim"

    def fit(self, train, dev):
        m = neural._get_model("minilm")
        self.male = m.encode(['he', 'man', 'boy', 'father', 'son', 'his', 'male'], normalize_embeddings=True)
        self.female = m.encode(['she', 'woman', 'girl', 'mother', 'daughter', 'her', 'female'], normalize_embeddings=True)
        return self

    def score(self, df):
        e = neural.embed(df.text, "minilm")
        return np.abs((e @ self.male.T).mean(1) - (e @ self.female.T).mean(1))


# ------------------------------------------------------------------ B3 zero-shot NLI
class ZeroShotNLI(System):
    name = "B3_nli_zeroshot"
    HYPOTHESIS = "This sentence expresses a stereotype about a social group."

    def __init__(self, model="facebook/bart-large-mnli", batch_size=32):
        self.model, self.bs = model, batch_size

    def fit(self, train, dev):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model)
        self.m = AutoModelForSequenceClassification.from_pretrained(self.model).to(self.device).eval()
        self.ent = [i for i, l in self.m.config.id2label.items() if l.lower().startswith("entail")][0]
        self.con = [i for i, l in self.m.config.id2label.items() if l.lower().startswith("contra")][0]
        return self

    def score(self, df):
        out = []
        texts = list(df.text)
        for i in range(0, len(texts), self.bs):
            b = texts[i:i + self.bs]
            enc = self.tok(b, [self.HYPOTHESIS] * len(b), return_tensors="pt", padding=True, truncation=True).to(self.device)
            with self.torch.no_grad():
                logits = self.m(**enc).logits[:, [self.con, self.ent]]
            out.extend(logits.softmax(-1)[:, 1].cpu().numpy())
        return np.array(out)


# ------------------------------------------------------------------ B4 fine-tuned transformer
class FineTuned(System):
    """B4: scores produced by experiments/train_transformer.py (run on Colab GPU)."""

    def __init__(self, model_key="roberta-base"):
        self.name = f"B4_ft_{model_key.replace('/', '_')}"
        self.key = model_key.replace("/", "_")

    def fit(self, train, dev):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.dir = ROOT / "results" / "finetuned" / f"{self.key}_s{self.seed}"
        if not self.dir.exists():
            raise FileNotFoundError(f"{self.dir} missing: run experiments/train_transformer.py first (Colab)")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.dir)
        self.m = AutoModelForSequenceClassification.from_pretrained(self.dir).to(self.device).eval()
        return self

    def score(self, df, bs=64):
        out, texts = [], list(df.text)
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
            with self.torch.no_grad():
                out.extend(self.m(**enc).logits.softmax(-1)[:, 1].cpu().numpy())
        return np.array(out)


# ------------------------------------------------------------------ B5 LLM judge
class LLMJudge(System):
    SYSTEM = (
        "You are an expert annotator of social bias in text. A text is STEREOTYPICAL if it associates "
        "a social group (gender, race/ethnicity/nationality/culture, religion, profession) with a trait, "
        "role, ability or behaviour in a generalizing way that matches a widespread stereotype "
        "(positive stereotypes count). Neutral mentions, factual statements, anti-stereotypes and "
        "explicit critiques of stereotypes are NOT stereotypical. "
        'Answer in JSON: {"stereotypical": 0 or 1, "confidence": number between 0 and 1, '
        '"category": "gender"|"origin"|"religion"|"profession"|"none", "rationale": short string}'
    )

    def __init__(self, provider=None, model=None):
        from biasdetect.llm.client import load_config
        cfg = load_config()["judge"]
        self.provider, self.model = provider or cfg["provider"], model or cfg["model"]
        self.name = f"B5_llm_{self.provider}_{self.model.replace('/', '_')}"

    def fit(self, train, dev):
        from biasdetect.llm.client import LLMClient
        self.client = LLMClient(self.provider, self.model)
        return self

    def judge(self, text: str) -> dict:
        return self.client.chat_json(self.SYSTEM, f"Text: {text}")

    def score(self, df):
        from tqdm import tqdm
        out = []
        for t in tqdm(df.text, desc=self.name):
            try:
                r = self.judge(t)
                y, c = int(r.get("stereotypical", 0)), float(r.get("confidence", 0.5))
                out.append(c if y else 1 - c)  # probability of 'stereotypical'
            except Exception as e:
                print("judge error:", e)
                out.append(np.nan)
        return np.array(out)


# ------------------------------------------------------------------ Proposed components
class SymbolicLR(System):
    """Ontology features only (ablation: symbolic branch)."""
    name = "P_sym"

    def fit(self, train, dev):
        self.fz = SymbolicFeaturizer(seed=self.seed)
        X = self.fz.transform_crossfit(train)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        self.clf.fit(X, train.label)
        return self

    def score(self, df):
        return self.clf.predict_proba(self.fz.transform(df.text))[:, 1]


class NeuralProbe(System):
    """Logistic probe on frozen sentence embeddings (ablation: neural branch)."""

    def __init__(self, enc="minilm", C=1.0):
        self.enc, self.C = enc, C
        self.name = f"P_neu_{enc}"

    def fit(self, train, dev):
        self.clf = LogisticRegression(max_iter=3000, C=self.C)
        self.clf.fit(neural.embed(train.text, self.enc), train.label)
        return self

    def score(self, df):
        return self.clf.predict_proba(neural.embed(df.text, self.enc))[:, 1]


class HybridFusion(System):
    """Proposed: learned late fusion (stacking).

    1. Neural probe scores on train are obtained by *group-wise cross-fitting*
       (no sample sees its own label) to avoid an over-confident neural feature.
    2. A logistic regression is trained on [logit(neural score) || ontology features].
    Coefficients give the learned weight of each branch (replaces the fixed 60/40).
    """

    def __init__(self, encs=("minilm",)):
        self.encs = list(encs)
        self.name = "P_fusion_" + "+".join(self.encs)

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def _neural_cols(self, df, train_mode=False, train=None):
        cols = []
        for enc in self.encs:
            X = neural.embed(df.text, enc)
            if train_mode:
                p = cross_val_predict(LogisticRegression(max_iter=3000), X, df.label,
                                      groups=df.group_id, cv=GroupKFold(5), method="predict_proba")[:, 1]
            else:
                p = self.probes[enc].predict_proba(X)[:, 1]
            cols.append(self._logit(p))
        return np.column_stack(cols)

    def fit(self, train, dev):
        self.fz = SymbolicFeaturizer(seed=self.seed)
        self.probes = {e: LogisticRegression(max_iter=3000).fit(neural.embed(train.text, e), train.label) for e in self.encs}
        X_sym = self.fz.transform_crossfit(train)
        X = np.hstack([self._neural_cols(train, train_mode=True), X_sym])
        self.feature_names = [f"neural_{e}" for e in self.encs] + list(SymbolicFeaturizer.FEATURE_NAMES)
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        self.clf.fit(X, train.label)
        return self

    def _X(self, df):
        return np.hstack([self._neural_cols(df), self.fz.transform(df.text)])

    def score(self, df):
        return self.clf.predict_proba(self._X(df))[:, 1]

    def branch_weights(self) -> pd.Series:
        coef = self.clf[-1].coef_[0]
        return pd.Series(coef, index=self.feature_names).sort_values(key=abs, ascending=False)

    def explain(self, text: str) -> dict:
        """Score + per-feature contributions + ontology rules fired (with sources)."""
        df = pd.DataFrame({"text": [text]})
        x = self._X(df)
        z = self.clf[0].transform(x)[0]
        contrib = pd.Series(z * self.clf[-1].coef_[0], index=self.feature_names)
        return {
            "score": float(self.clf.predict_proba(x)[0, 1]),
            "contributions": contrib[contrib.abs() > 1e-6].sort_values(key=abs, ascending=False).round(3).to_dict(),
            "rules": [r.__dict__ for r in self.fz.analyze(text).rules],
        }


def registry(include_slow=False, include_llm=False):
    systems = [Majority(), LegacyRules(), LegacyGenderSim(), SymbolicLR()]
    systems += [NeuralProbe(e) for e in neural.ENCODERS]
    systems += [HybridFusion([e]) for e in neural.ENCODERS]
    systems += [HybridFusion(list(neural.ENCODERS))]
    if include_slow:
        systems += [ZeroShotNLI()]
    if include_llm:
        systems += [LLMJudge()]
    return systems
