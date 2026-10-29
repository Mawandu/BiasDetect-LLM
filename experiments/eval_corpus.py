"""Evaluate the score files produced by experiments/score_corpus.py.

Part B only: no model is trained here. The script reads the corpus and the
per-system/per-mode/per-seed score files, then writes metrics, paired
differences and a run-info file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from biasdetect.eval.metrics import mcnemar
from experiments.score_corpus import MODES, git_commit, verify_sha256sums

ROOT = Path(__file__).resolve().parents[1]
SCORE_RE = re.compile(r"(.+)__(full|sentmean)__s(\d+)\.csv")
SUBSET_ORDER = ("all", "core", "other", "all_norefusal", "core_norefusal", "other_norefusal")
GOLD_ORDER = ("gold_raw", "gold_reviewed", "gold_A1A2", "gold_raw_nw15")
METRIC_ORDER = ("auroc", "macro_f1", "ece")
MAIN_SUBSET = "core_norefusal"
MAIN_GOLD = "gold_raw"
DEFAULT_EXPECT_SEEDS = (13, 21, 42, 7, 99)
PAIRS = (
    ("P_fusion_minilm", "P_neu_minilm"),
    ("P_fusion_minilm", "P_sym"),
)
CORPUS_COLUMNS = {
    "id", "prompt_id", "is_core", "refusal", "gold_raw", "gold_reviewed",
    "A1_round1", "A2_round1", "n_words",
}
SCORE_COLUMNS = {"id", "prompt_id", "score", "threshold"}


@dataclass(frozen=True)
class EvalSpec:
    subset: str
    gold: str
    idx: np.ndarray
    y: np.ndarray
    prompt_ids: np.ndarray


def holm(pvalues) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(pvalues[finite], kind="stable")]
    previous = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(order) - rank) * pvalues[index])
        previous = max(previous, value)
        adjusted[index] = previous
    return adjusted


def _auroc(y, scores) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    positive = y == 1
    negative = y == 0
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    starts = ends - counts + 1
    average_ranks = (starts + ends) / 2
    ranks = average_ranks[inverse]
    return float(
        (ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _macro_f1(y, predictions) -> float:
    y = np.asarray(y)
    predictions = np.asarray(predictions, dtype=bool)
    tp = int(np.sum(predictions & (y == 1)))
    fp = int(np.sum(predictions & (y == 0)))
    fn = int(np.sum(~predictions & (y == 1)))
    denom = 2 * tp + fp + fn
    f1_pos = 2 * tp / denom if denom else 0.0

    tn = int(np.sum(~predictions & (y == 0)))
    fp_neg = int(np.sum(~predictions & (y == 1)))
    fn_neg = int(np.sum(predictions & (y == 0)))
    denom_neg = 2 * tn + fp_neg + fn_neg
    f1_neg = 2 * tn / denom_neg if denom_neg else 0.0
    return float((f1_pos + f1_neg) / 2)


def _ece(y, scores, bins=10) -> float:
    y = np.asarray(y, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, bins - 1)
    counts = np.bincount(idx, minlength=bins)
    sum_y = np.bincount(idx, weights=y, minlength=bins)
    sum_scores = np.bincount(idx, weights=scores, minlength=bins)
    nonempty = counts > 0
    gaps = np.abs(
        sum_y[nonempty] / counts[nonempty] - sum_scores[nonempty] / counts[nonempty]
    )
    return float(np.sum(gaps * counts[nonempty] / len(y)))


def compute_metrics(
    y, scores, threshold, ece_valid: bool | None = None
) -> dict[str, float | str]:
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"auroc": np.nan, "macro_f1": np.nan, "ece": np.nan}
    auroc = _auroc(y, scores)
    macro_f1 = _macro_f1(y, scores >= threshold)
    if ece_valid is None:
        ece_valid = scores.min() >= 0 and scores.max() <= 1
    calibration: float | str = _ece(y, scores) if ece_valid else "n/a"
    return {
        "auroc": float(auroc),
        "macro_f1": float(macro_f1),
        "ece": calibration,
    }


def bootstrap_prompt_ids(prompt_ids, rng) -> np.ndarray:
    prompt_ids = np.asarray(prompt_ids)
    universe = pd.unique(prompt_ids)
    if len(universe) == 0:
        return np.array([], dtype=prompt_ids.dtype)
    return rng.choice(universe, size=len(universe), replace=True)


def draw_cluster_indices(drawn_prompts, groups) -> np.ndarray:
    selected = [groups[prompt] for prompt in drawn_prompts if prompt in groups]
    return np.concatenate(selected) if selected else np.array([], dtype=int)


def _bootstrap_p_value(diffs) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return 1.0
    b = len(diffs)
    k_pos = int(np.sum(diffs >= 0))
    k_neg = int(np.sum(diffs <= 0))
    return float(min(1.0, 2.0 * (min(k_pos, k_neg) + 1) / (b + 1)))


def _nanmean(values) -> float | str:
    numeric = []
    has_na = False
    for value in values:
        if isinstance(value, str):
            has_na = has_na or value == "n/a"
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    if numeric:
        return float(np.mean(numeric))
    return "n/a" if has_na else np.nan


def _interval(values) -> tuple[float | str, float | str]:
    numeric = []
    has_na = False
    for value in values:
        if isinstance(value, str):
            has_na = has_na or value == "n/a"
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    if numeric:
        low, high = np.percentile(numeric, [2.5, 97.5])
        return float(low), float(high)
    if has_na:
        return "n/a", "n/a"
    return np.nan, np.nan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run_info(preds_dir: Path, corpus_sha256: str | None = None):
    paths = sorted(preds_dir.glob("RUN_INFO*.json"))
    if not paths:
        return None
    systems_seen = None
    seeds_seen: set[int] = set()
    for path in paths:
        try:
            info = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"RUN_INFO invalide : {path}") from exc
        actual = info.get("corpus_sha256")
        if corpus_sha256 is not None and actual != corpus_sha256:
            raise ValueError(
                f"hash du corpus incohérent dans {path.name} : "
                f"attendu {corpus_sha256}, obtenu {actual}"
            )
        systems = info.get("systems")
        seeds = info.get("seeds")
        if not isinstance(systems, (list, tuple)) or not isinstance(seeds, (list, tuple)):
            raise ValueError(f"RUN_INFO invalide : {path}")
        if not systems or not seeds:
            raise ValueError(f"RUN_INFO invalide : {path}")
        current_systems = sorted(str(system) for system in systems)
        if systems_seen is None:
            systems_seen = current_systems
        elif current_systems != systems_seen:
            raise ValueError(
                f"systems incohérents entre RUN_INFO : {current_systems} != {systems_seen}"
            )
        for seed in seeds:
            seed = int(seed)
            if seed in seeds_seen:
                raise ValueError(f"seed {seed} présente dans plusieurs RUN_INFO")
            seeds_seen.add(seed)
    return systems_seen, sorted(seeds_seen), paths


def discover_score_files(
    preds_dir: Path,
    corpus_sha256: str | None = None,
    expect_seeds=DEFAULT_EXPECT_SEEDS,
):
    preds_dir = Path(preds_dir)
    manifest = _read_run_info(preds_dir, corpus_sha256)
    files = sorted(preds_dir.glob("*.csv")) if preds_dir.exists() else []
    if manifest is None:
        if not files:
            raise FileNotFoundError(f"aucun fichier de scores dans {preds_dir}")
        systems: set[str] = set()
        seeds: set[int] = set()
        for path in files:
            match = SCORE_RE.fullmatch(path.name)
            if not match:
                raise ValueError(f"nom de fichier de scores inattendu : {path.name}")
            systems.add(match.group(1))
            seeds.add(int(match.group(3)))
        systems, seeds = sorted(systems), sorted(seeds)
        run_info_paths = []
    else:
        systems, seeds, run_info_paths = manifest

    if expect_seeds is not None:
        expected_seeds = sorted(int(seed) for seed in expect_seeds)
        if seeds != expected_seeds:
            raise ValueError(
                f"seeds inattendues (--expect-seeds {expected_seeds}) : obtenu {seeds}"
            )

    expected = [
        preds_dir / f"{system}__{mode}__s{seed}.csv"
        for system in systems
        for mode in MODES
        for seed in seeds
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        listed = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"fichier(s) de scores manquant(s) : {listed}")

    expected_names = {path.name for path in expected}
    extra = [path for path in files if path.name not in expected_names]
    if extra:
        listed = ", ".join(str(path) for path in extra[:5])
        raise ValueError(f"fichier(s) de scores non attendu(s) : {listed}")
    return systems, seeds, run_info_paths, expected


def _check_pair_systems(systems) -> None:
    required = {name for pair in PAIRS for name in pair}
    missing = required - set(systems)
    if missing:
        raise FileNotFoundError(
            f"système(s) requis pour les comparaisons appariées absent(s) : {sorted(missing)}"
        )


def _read_corpus(corpus_path: Path) -> pd.DataFrame:
    corpus = pd.read_csv(corpus_path)
    missing = CORPUS_COLUMNS - set(corpus.columns)
    if missing:
        raise ValueError(f"colonnes manquantes dans le corpus : {sorted(missing)}")
    if corpus["id"].duplicated().any():
        raise ValueError("identifiants dupliqués dans le corpus")
    if corpus["prompt_id"].isna().any():
        raise ValueError("prompt_id manquant dans le corpus")
    return corpus


def _permute_gold(corpus: pd.DataFrame, seed: int) -> pd.DataFrame:
    order = np.random.default_rng(seed).permutation(len(corpus))
    permuted = corpus.copy()
    for column in ("gold_raw", "gold_reviewed", "A1_round1", "A2_round1"):
        permuted[column] = permuted[column].to_numpy()[order]
    return permuted


def _load_scores(preds_dir: Path, systems, seeds, corpus: pd.DataFrame):
    scores = {}
    thresholds = {}
    for system in systems:
        for mode in MODES:
            for seed in seeds:
                path = preds_dir / f"{system}__{mode}__s{seed}.csv"
                frame = pd.read_csv(path)
                missing = SCORE_COLUMNS - set(frame.columns)
                if missing:
                    raise ValueError(f"colonnes manquantes dans {path.name} : {sorted(missing)}")
                if frame["id"].duplicated().any():
                    raise ValueError(f"identifiants dupliqués dans {path.name}")
                if len(frame) != len(corpus) or set(frame["id"]) != set(corpus["id"]):
                    raise ValueError(f"identifiants incohérents dans {path.name}")

                merged = corpus[["id", "prompt_id"]].merge(
                    frame[["id", "prompt_id", "score", "threshold"]],
                    on="id",
                    how="left",
                    suffixes=("", "_score"),
                    validate="one_to_one",
                )
                if merged["score"].isna().any():
                    raise ValueError(f"scores manquants dans {path.name}")
                if not (merged["prompt_id"] == merged["prompt_id_score"]).all():
                    raise ValueError(f"prompt_id incohérent dans {path.name}")
                if merged["threshold"].nunique(dropna=False) != 1:
                    raise ValueError(f"seuil non constant dans {path.name}")

                score = merged["score"].to_numpy(dtype=float)
                threshold = float(merged["threshold"].iloc[0])
                if not np.isfinite(score).all() or not np.isfinite(threshold):
                    raise ValueError(f"valeur non finie dans {path.name}")
                scores[(system, mode, seed)] = score
                thresholds[(system, mode, seed)] = threshold
    return scores, thresholds


def _subsets(corpus: pd.DataFrame) -> dict[str, np.ndarray]:
    core = corpus["is_core"].fillna(False).astype(bool)
    refusal = corpus["refusal"].fillna(False).astype(bool)
    return {
        "all": np.ones(len(corpus), dtype=bool),
        "core": core.to_numpy(),
        "other": (~core).to_numpy(),
        "all_norefusal": (~refusal).to_numpy(),
        "core_norefusal": (core & ~refusal).to_numpy(),
        "other_norefusal": (~core & ~refusal).to_numpy(),
    }


def _gold_masks(corpus: pd.DataFrame, subset_mask: np.ndarray):
    raw = pd.to_numeric(corpus["gold_raw"], errors="coerce")
    reviewed = pd.to_numeric(corpus["gold_reviewed"], errors="coerce")
    a1 = pd.to_numeric(corpus["A1_round1"], errors="coerce")
    a2 = pd.to_numeric(corpus["A2_round1"], errors="coerce")
    n_words = pd.to_numeric(corpus["n_words"], errors="coerce")
    raw_ok = raw.isin([0, 1]).to_numpy()
    reviewed_ok = reviewed.isin([0, 1]).to_numpy()
    a1a2_ok = (a1.isin([0, 1]) & a2.isin([0, 1]) & (a1 == a2)).to_numpy()
    return {
        "gold_raw": (subset_mask & raw_ok, raw),
        "gold_reviewed": (subset_mask & reviewed_ok, reviewed),
        "gold_A1A2": (subset_mask & a1a2_ok, a1),
        "gold_raw_nw15": (subset_mask & raw_ok & (n_words >= 15).to_numpy(), raw),
    }


def _make_specs(corpus: pd.DataFrame) -> dict[tuple[str, str], EvalSpec]:
    subset_masks = _subsets(corpus)
    specs = {}
    for subset in SUBSET_ORDER:
        golds = GOLD_ORDER if subset == MAIN_SUBSET else (MAIN_GOLD,)
        gold_masks = _gold_masks(corpus, subset_masks[subset])
        for gold in golds:
            selected, labels = gold_masks[gold]
            idx = np.flatnonzero(selected)
            y = labels.to_numpy()[idx].astype(int)
            prompt_ids = corpus["prompt_id"].to_numpy()[idx]
            specs[(subset, gold)] = EvalSpec(subset, gold, idx, y, prompt_ids)
    return specs


def _seed_metrics(
    spec: EvalSpec, system, mode, seed, scores, thresholds, ece_valid=None
):
    return compute_metrics(
        spec.y,
        scores[(system, mode, seed)][spec.idx],
        thresholds[(system, mode, seed)],
        ece_valid=ece_valid,
    )


def evaluate(corpus, systems, seeds, scores, thresholds, n_boot, seed):
    specs = _make_specs(corpus)
    subset_masks = _subsets(corpus)
    prompt_column = corpus["prompt_id"].to_numpy()
    subset_prompts = {
        subset: pd.unique(prompt_column[mask])
        for subset, mask in subset_masks.items()
    }
    groups = {
        key: {
            prompt: np.flatnonzero(spec.prompt_ids == prompt)
            for prompt in pd.unique(spec.prompt_ids)
        }
        for key, spec in specs.items()
    }
    specs_by_subset = {subset: [] for subset in SUBSET_ORDER}
    for key, spec in specs.items():
        specs_by_subset[spec.subset].append((key, spec))

    ece_valid = {}
    for key, spec in specs.items():
        for system in systems:
            for mode in MODES:
                valid = True
                for eval_seed in seeds:
                    full_scores = scores[(system, mode, eval_seed)][spec.idx]
                    if len(full_scores) and (
                        full_scores.min() < 0 or full_scores.max() > 1
                    ):
                        valid = False
                        break
                ece_valid[(system, mode, spec.subset, spec.gold)] = valid

    metric_boot: dict[tuple, list[float | str]] = {}
    n_boot_rejected = {key: 0 for key in specs}
    paired_boot = {(pair, mode): [] for pair in PAIRS for mode in MODES}

    for subset in SUBSET_ORDER:
        universe = subset_prompts[subset]
        subset_specs = specs_by_subset[subset]
        if len(universe) == 0:
            for key, _ in subset_specs:
                n_boot_rejected[key] = n_boot
            continue

        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            drawn = bootstrap_prompt_ids(universe, rng)
            for key, spec in subset_specs:
                idx = draw_cluster_indices(drawn, groups[key])
                y_boot = spec.y[idx]
                if len(np.unique(y_boot)) < 2:
                    n_boot_rejected[key] += 1
                    continue

                current = {}
                for system in systems:
                    for mode in MODES:
                        seed_values = {metric: [] for metric in METRIC_ORDER}
                        for eval_seed in seeds:
                            values = compute_metrics(
                                y_boot,
                                scores[(system, mode, eval_seed)][spec.idx][idx],
                                thresholds[(system, mode, eval_seed)],
                                ece_valid=ece_valid[
                                    (system, mode, spec.subset, spec.gold)
                                ],
                            )
                            for metric in METRIC_ORDER:
                                seed_values[metric].append(values[metric])
                        for metric in METRIC_ORDER:
                            metric_boot.setdefault(
                                (system, mode, spec.subset, spec.gold, metric), []
                            ).append(_nanmean(seed_values[metric]))
                        if key == (MAIN_SUBSET, MAIN_GOLD):
                            current[(system, mode)] = seed_values

                if key == (MAIN_SUBSET, MAIN_GOLD):
                    for pair in PAIRS:
                        for mode in MODES:
                            diffs = [
                                current[(pair[0], mode)]["auroc"][i]
                                - current[(pair[1], mode)]["auroc"][i]
                                for i in range(len(seeds))
                            ]
                            paired_boot[(pair, mode)].append(_nanmean(diffs))

    metric_records = []
    for key, spec in specs.items():
        n = len(spec.y)
        n_prompts_subset = int(len(pd.unique(spec.prompt_ids)))
        n_prompts_boot = int(len(pd.unique(subset_prompts[spec.subset])))
        rejected = n_boot_rejected[key]
        for system in systems:
            for mode in MODES:
                per_seed = [
                    _seed_metrics(
                        spec, system, mode, eval_seed, scores, thresholds,
                        ece_valid=ece_valid[
                            (system, mode, spec.subset, spec.gold)
                        ],
                    )
                    for eval_seed in seeds
                ]
                for metric in METRIC_ORDER:
                    estimate = _nanmean([values[metric] for values in per_seed])
                    boot = metric_boot.get(
                        (system, mode, spec.subset, spec.gold, metric), []
                    )
                    ci_low, ci_high = _interval(boot)
                    metric_records.append({
                        "system": system,
                        "mode": mode,
                        "subset": spec.subset,
                        "gold": spec.gold,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n": n,
                        "n_prompts_subset": n_prompts_subset,
                        "n_prompts_boot": n_prompts_boot,
                        "n_seeds": len(seeds),
                        "n_boot": n_boot,
                        "n_boot_rejected": rejected,
                    })

    paired_records = []
    main = specs[(MAIN_SUBSET, MAIN_GOLD)]
    main_rejected = n_boot_rejected[(MAIN_SUBSET, MAIN_GOLD)]
    main_prompts_subset = int(len(pd.unique(main.prompt_ids)))
    main_prompts_boot = int(len(pd.unique(subset_prompts[MAIN_SUBSET])))
    for pair in PAIRS:
        for mode in MODES:
            diffs = np.asarray(paired_boot[(pair, mode)], dtype=float)
            diffs = diffs[np.isfinite(diffs)]
            if len(diffs):
                ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
            else:
                ci_low = ci_high = np.nan
            p_value = _bootstrap_p_value(diffs)

            point_diffs = [
                _seed_metrics(main, pair[0], mode, eval_seed, scores, thresholds)["auroc"]
                - _seed_metrics(main, pair[1], mode, eval_seed, scores, thresholds)["auroc"]
                for eval_seed in seeds
            ]
            record = {
                "comparison": f"{pair[0]}-{pair[1]}",
                "mode": mode,
                "subset": MAIN_SUBSET,
                "gold": MAIN_GOLD,
                "metric": "auroc",
                "difference": _nanmean(point_diffs),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "p": float(p_value),
                "n_boot": n_boot,
                "n_boot_rejected": main_rejected,
                "n_prompts_subset": main_prompts_subset,
                "n_prompts_boot": main_prompts_boot,
            }
            for eval_seed in seeds:
                pred_a = (
                    scores[(pair[0], mode, eval_seed)][main.idx]
                    >= thresholds[(pair[0], mode, eval_seed)]
                )
                pred_b = (
                    scores[(pair[1], mode, eval_seed)][main.idx]
                    >= thresholds[(pair[1], mode, eval_seed)]
                )
                record[f"mcnemar_p_s{eval_seed}"] = mcnemar(main.y, pred_a, pred_b)
            paired_records.append(record)

    adjusted = holm([record["p"] for record in paired_records])
    for record, corrected in zip(paired_records, adjusted):
        record["p_holm"] = float(corrected)

    metric_columns = [
        "system", "mode", "subset", "gold", "metric", "estimate",
        "ci_low", "ci_high", "n", "n_prompts_subset", "n_prompts_boot",
        "n_seeds", "n_boot", "n_boot_rejected",
    ]
    paired_columns = [
        "comparison", "mode", "subset", "gold", "metric", "difference",
        "ci_low", "ci_high", "p", "p_holm", "n_boot", "n_boot_rejected",
        "n_prompts_subset", "n_prompts_boot",
    ] + [f"mcnemar_p_s{eval_seed}" for eval_seed in seeds]

    metrics_df = pd.DataFrame(metric_records, columns=metric_columns)
    metrics_df = metrics_df.sort_values(
        ["subset", "gold", "system", "mode", "metric"], kind="stable"
    ).reset_index(drop=True)
    paired_df = pd.DataFrame(paired_records, columns=paired_columns)
    paired_df = paired_df.sort_values(
        ["comparison", "mode"], kind="stable"
    ).reset_index(drop=True)
    return metrics_df, paired_df


def run(
    preds_dir,
    corpus_path,
    out_dir,
    n_boot=2000,
    seed=13,
    expect_seeds=DEFAULT_EXPECT_SEEDS,
    overwrite=False,
    permute_gold=None,
):
    preds_dir = Path(preds_dir)
    corpus_path = Path(corpus_path)
    out_dir = Path(out_dir)
    if n_boot < 1:
        raise ValueError("--boot doit être strictement positif")
    if permute_gold is not None and out_dir.resolve() == (ROOT / "results").resolve():
        raise ValueError("--permute-gold refuse le dossier results par défaut")

    now = datetime.now()
    info_path = out_dir / f"corpus_eval_INFO_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    metrics_path = out_dir / "corpus_metrics.csv"
    paired_path = out_dir / "corpus_paired.csv"
    if not overwrite:
        existing = [
            path for path in (metrics_path, paired_path, info_path) if path.exists()
        ]
        if existing:
            listed = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"fichier(s) déjà présent(s), utilisez --overwrite : {listed}"
            )

    verify_sha256sums(corpus_path.parent)
    corpus_sha256 = _sha256(corpus_path)
    systems, seeds, run_info_paths, score_files = discover_score_files(
        preds_dir, corpus_sha256, expect_seeds
    )
    _check_pair_systems(systems)
    verify_sha256sums(preds_dir)
    corpus = _read_corpus(corpus_path)
    if permute_gold is not None:
        corpus = _permute_gold(corpus, permute_gold)
    scores, thresholds = _load_scores(preds_dir, systems, seeds, corpus)

    metrics_df, paired_df = evaluate(
        corpus, systems, seeds, scores, thresholds, n_boot, seed
    )

    info = {
        "date": now.isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seed": seed,
        "seeds": seeds,
        "n_boot": n_boot,
        "corpus_sha256": corpus_sha256,
        "preds_sha256sums_sha256": _sha256(preds_dir / "SHA256SUMS"),
        "run_info_files": [path.name for path in run_info_paths],
        "n_score_files": len(score_files),
        "git_commit": git_commit(),
        "systems": systems,
        "modes": list(MODES),
        "subsets": list(SUBSET_ORDER),
        "golds": list(GOLD_ORDER),
        "permuted": permute_gold is not None,
        "permute_gold": permute_gold,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return metrics_df, paired_df, info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate corpus score files (part B).")
    parser.add_argument(
        "--preds", type=Path, default=ROOT / "results" / "corpus_preds"
    )
    parser.add_argument(
        "--corpus", type=Path, default=ROOT / "corpus" / "biasdetect_corpus.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--expect-seeds", type=int, nargs="+", default=list(DEFAULT_EXPECT_SEEDS)
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--permute-gold", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        args.preds,
        args.corpus,
        args.out,
        args.boot,
        args.seed,
        expect_seeds=args.expect_seeds,
        overwrite=args.overwrite,
        permute_gold=args.permute_gold,
    )


if __name__ == "__main__":
    main()
