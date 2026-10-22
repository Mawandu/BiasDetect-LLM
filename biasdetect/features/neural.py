"""Frozen sentence encoders with on-disk embedding cache."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"

# Encoders compared in the paper (all public, sentence-transformers compatible)
ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "e5": "intfloat/e5-base-v2",
    "bge": "BAAI/bge-base-en-v1.5",
}
# Some models expect an instruction prefix
PREFIX = {"e5": "query: "}

_models = {}


def _get_model(key: str):
    from sentence_transformers import SentenceTransformer

    if key not in _models:
        _models[key] = SentenceTransformer(ENCODERS[key])
    return _models[key]


def embed(texts, key: str, batch_size: int = 64) -> np.ndarray:
    texts = [PREFIX.get(key, "") + str(t) for t in texts]
    h = hashlib.md5(("\n".join(texts)).encode()).hexdigest()[:16]
    path = CACHE / f"{key}_{h}.npy"
    if path.exists():
        return np.load(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    emb = _get_model(key).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 500)
    np.save(path, emb)
    return emb
