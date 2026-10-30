"""Score the BiasDetect corpus with the trained systems.

Part A only: this script produces per-system, per-mode, per-seed score files.
Metrics and bootstrap intervals are deliberately left to a separate step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from biasdetect.eval.metrics import best_threshold
from biasdetect.models import systems as S
from experiments.run import eval_sets

ROOT = Path(__file__).resolve().parents[1]
MODES = ("full", "sentmean")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    parts = [p for p in SENTENCE_RE.split(str(text)) if p.strip()]
    return parts or [str(text)]


def score_texts(sysobj, texts, mode: str) -> np.ndarray:
    texts = [str(t) for t in texts]
    if mode == "full":
        if not texts:
            return np.array([], dtype=float)
        return np.asarray(sysobj.score(pd.DataFrame({"text": texts})), dtype=float)
    if mode != "sentmean":
        raise ValueError(f"mode inconnu : {mode!r}")

    pieces: list[str] = []
    owners: list[int] = []
    for i, text in enumerate(texts):
        sentences = split_sentences(text)
        pieces.extend(sentences)
        owners.extend([i] * len(sentences))
    if not pieces:
        return np.array([], dtype=float)

    scores = np.asarray(sysobj.score(pd.DataFrame({"text": pieces})), dtype=float)
    sums = np.zeros(len(texts), dtype=float)
    counts = np.zeros(len(texts), dtype=float)
    for owner, score in zip(owners, scores):
        sums[owner] += score
        counts[owner] += 1
    return sums / counts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256sums(corpus_dir: Path) -> dict[str, str]:
    corpus_dir = Path(corpus_dir)
    sums_path = corpus_dir / "SHA256SUMS"
    if not sums_path.exists():
        raise FileNotFoundError(f"SHA256SUMS introuvable : {sums_path}")

    checked: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"ligne SHA256SUMS invalide : {line!r}")
        expected, name = parts[0].lower(), parts[1].lstrip("*").strip()
        path = corpus_dir / name
        if not path.exists():
            raise FileNotFoundError(f"fichier listé dans SHA256SUMS introuvable : {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"sha256 invalide pour {path} : attendu {expected}, obtenu {actual}"
            )
        checked[name] = actual
    return checked


def git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "non vérifié"
    return proc.stdout.strip()


def _select_systems(only: list[str] | None):
    systems = S.registry()
    if only:
        wanted = set(only)
        systems = [s for s in systems if s.name in wanted]
        missing = wanted - {s.name for s in systems}
        if missing:
            raise ValueError(f"système(s) inconnu(s) dans --only : {sorted(missing)}")
    if not systems:
        raise ValueError("aucun système sélectionné")
    return systems


def run(
    seeds,
    corpus_path,
    out_dir,
    only: list[str] | None = None,
    overwrite: bool = False,
) -> dict:
    seeds = [int(seed) for seed in seeds]
    corpus_path = Path(corpus_path)
    out_dir = Path(out_dir)

    verify_sha256sums(corpus_path.parent)
    corpus = pd.read_csv(corpus_path)
    corpus_sha = _sha256(corpus_path)

    selected = _select_systems(only)
    names = [sysobj.name for sysobj in selected]
    targets = [
        out_dir / f"{name}__{mode}__s{seed}.csv"
        for seed in seeds
        for name in names
        for mode in MODES
    ]
    if not overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                f"{len(existing)} fichier(s) déjà présent(s), utilisez --overwrite : {existing[0]}"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    texts = corpus["text"].astype(str).tolist()

    for seed in seeds:
        train, dev, _ = eval_sets(seed, None, False)
        for sysobj in _select_systems(only):
            sysobj.seed = seed
            sysobj.fit(train, dev)
            dev_scores = sysobj.score(dev)
            threshold = best_threshold(dev.label.values, dev_scores)
            for mode in MODES:
                scores = score_texts(sysobj, texts, mode)
                out = pd.DataFrame({
                    "id": corpus["id"],
                    "prompt_id": corpus["prompt_id"],
                    "score": scores,
                    "threshold": threshold,
                })
                out.to_csv(out_dir / f"{sysobj.name}__{mode}__s{seed}.csv", index=False)

    info = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seeds": seeds,
        "corpus_sha256": corpus_sha,
        "git_commit": git_commit(),
        "systems": names,
    }
    (out_dir / f"RUN_INFO_{datetime.now():%Y%m%d-%H%M%S}.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n"
    )
    return info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Score the corpus (part A).")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 21, 42, 7, 99])
    parser.add_argument(
        "--corpus",
        type=Path,
        default=ROOT / "corpus" / "biasdetect_corpus.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "corpus_preds",
    )
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(args.seeds, args.corpus, args.out, args.only, args.overwrite)


if __name__ == "__main__":
    main()
