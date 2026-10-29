"""Evaluate the benchmark score files (StereoSet, CrowS-Pairs, SBIC).

The script is descriptive: it reads existing score files, computes metrics and
cluster-bootstrap intervals, and writes metrics, paired differences and a
run-info file. No model is trained and no threshold is tuned here.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from biasdetect.eval.metrics import pairwise_accuracy
from experiments.eval_corpus import (
    DEFAULT_EXPECT_SEEDS,
    _auroc,
    _bootstrap_p_value,
    _interval,
    _macro_f1,
    _nanmean,
    _sha256,
    bootstrap_prompt_ids,
    draw_cluster_indices,
    holm,
)
from experiments.score_corpus import git_commit

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("stereoset_test", "crows", "sbic")
PAIR_SYSTEMS = ("P_fusion_minilm", "P_neu_minilm", "P_sym")
PAIRS = (
    ("P_fusion_minilm", "P_neu_minilm"),
    ("P_fusion_minilm", "P_sym"),
)
PAIR_DATASETS = ("stereoset_test", "crows")
BASE_METRICS = ("auroc", "macro_f1")
ITEM_COLUMNS = ("text", "label", "category", "group_id", "pair_id")
REQUIRED_COLUMNS = {"text", "label", "category", "group_id", "score", "threshold"}
PRED_RE = re.compile(
    r"(.+)__(stereoset_test|crows|sbic)__s(\d+)\.csv"
)


@dataclass(frozen=True)
class BenchmarkData:
    frames: dict
    scores: dict
    thresholds: dict
    systems: tuple[str, ...]
    seeds_by_system: dict[str, tuple[int, ...]]
    files: tuple[Path, ...]
    identical_by_dataset: dict[str, bool]


def _metrics_for_dataset(dataset: str) -> tuple[str, ...]:
    return BASE_METRICS + (("pair_acc",) if dataset in PAIR_DATASETS else ())


def compute_metrics(frame: pd.DataFrame, scores, threshold) -> dict[str, float]:
    """Compute AUROC, macro-F1 and (when available) pairwise accuracy."""
    y = frame["label"].to_numpy(dtype=int)
    scores = np.asarray(scores, dtype=float)
    values = {"auroc": np.nan, "macro_f1": np.nan}
    if len(y) and len(np.unique(y)) >= 2:
        values["auroc"] = float(_auroc(y, scores))
        values["macro_f1"] = float(_macro_f1(y, scores >= threshold))
    if "pair_id" in frame.columns:
        values["pair_acc"] = float(
            pairwise_accuracy(frame.assign(group_id=frame["pair_id"]), scores)
        )
    return values


def _validate_frame(frame: pd.DataFrame, path: Path, dataset: str) -> None:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"colonnes manquantes dans {path.name} : {sorted(missing)}")
    if frame["label"].isna().any() or not frame["label"].isin([0, 1]).all():
        raise ValueError(f"labels invalides dans {path.name}")
    if frame["group_id"].isna().any():
        raise ValueError(f"group_id manquant dans {path.name}")
    if frame["text"].isna().any():
        raise ValueError(f"texte manquant dans {path.name}")
    if frame["score"].isna().any() or not np.isfinite(frame["score"]).all():
        raise ValueError(f"score invalide dans {path.name}")
    if frame["threshold"].nunique(dropna=False) != 1:
        raise ValueError(f"seuil non constant dans {path.name}")
    if not np.isfinite(frame["threshold"].iloc[0]):
        raise ValueError(f"seuil invalide dans {path.name}")

    if dataset in PAIR_DATASETS and "pair_id" not in frame.columns:
        raise ValueError(f"pair_id manquant dans {path.name}")
    if "pair_id" in frame.columns:
        if frame["pair_id"].isna().any():
            raise ValueError(f"pair_id manquant dans {path.name}")
        per_pair_groups = frame.groupby("pair_id")["group_id"].nunique()
        if (per_pair_groups != 1).any():
            raise ValueError(f"pair_id rattaché à plusieurs group_id dans {path.name}")
        per_pair_sizes = frame.groupby("pair_id").size()
        if (per_pair_sizes != 2).any():
            raise ValueError(f"pair_id incomplet dans {path.name}")
        per_pair_labels = frame.groupby("pair_id")["label"].apply(
            lambda values: set(int(value) for value in values)
        )
        if not per_pair_labels.map(lambda values: values == {0, 1}).all():
            raise ValueError(f"pair_id sans les deux labels dans {path.name}")


def discover_score_files(preds_dir, expect_seeds=DEFAULT_EXPECT_SEEDS):
    """Return systems, seeds per system and the list of score files."""
    preds_dir = Path(preds_dir)
    files = sorted(preds_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"aucun fichier de scores dans {preds_dir}")

    seeds_by_system_dataset: dict[str, dict[str, set[int]]] = {}
    for path in files:
        match = PRED_RE.fullmatch(path.name)
        if not match:
            raise ValueError(f"nom de fichier de scores inattendu : {path.name}")
        system, dataset, seed = match.group(1), match.group(2), int(match.group(3))
        seeds_by_system_dataset.setdefault(system, {}).setdefault(
            dataset, set()
        ).add(seed)

    systems = tuple(sorted(seeds_by_system_dataset))
    seeds_by_system: dict[str, tuple[int, ...]] = {}
    for system in systems:
        by_dataset = seeds_by_system_dataset[system]
        missing = [dataset for dataset in DATASETS if dataset not in by_dataset]
        if missing:
            raise FileNotFoundError(
                f"jeu(x) manquant(s) pour {system} : {missing}"
            )
        seed_sets = [by_dataset[dataset] for dataset in DATASETS]
        if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
            raise ValueError(f"seeds incohérentes entre jeux pour {system}")
        seeds_by_system[system] = tuple(sorted(seed_sets[0]))

    missing_systems = [system for system in PAIR_SYSTEMS if system not in systems]
    if missing_systems:
        raise FileNotFoundError(
            "système(s) requis pour les comparaisons appariées absent(s) : "
            f"{missing_systems}"
        )
    p_seed_sets = [set(seeds_by_system[system]) for system in PAIR_SYSTEMS]
    if any(seed_set != p_seed_sets[0] for seed_set in p_seed_sets[1:]):
        raise ValueError(
            "seeds incohérentes entre les systèmes P : "
            f"{[sorted(seed_set) for seed_set in p_seed_sets]}"
        )
    if expect_seeds is not None:
        expected = sorted(int(seed) for seed in expect_seeds)
        observed = sorted(p_seed_sets[0])
        if observed != expected:
            raise ValueError(
                "seeds inattendues pour les systèmes P "
                f"(--expect-seeds {expected}) : obtenu {observed}"
            )
    return systems, seeds_by_system, tuple(files)


def _item_signature(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ITEM_COLUMNS if column in frame.columns]
    return frame[columns].reset_index(drop=True)


def _validate_cross_system_items(frames, systems, seeds_by_system) -> None:
    for dataset in DATASETS:
        seeds = sorted({
            seed
            for system in systems
            for seed in seeds_by_system[system]
        })
        for seed in seeds:
            signatures = []
            for system in systems:
                if seed not in seeds_by_system[system]:
                    continue
                frame = frames[(system, dataset, seed)]
                signatures.append((system, _item_signature(frame)))
            base_system, base = signatures[0]
            for system, signature in signatures[1:]:
                if not base.equals(signature):
                    raise ValueError(
                        "items incohérents entre systèmes pour "
                        f"{dataset} s{seed} : {base_system} != {system}"
                    )


def _identical_by_dataset(frames, systems, seeds_by_system) -> dict[str, bool]:
    identical = {}
    for dataset in DATASETS:
        seeds = sorted({
            seed
            for system in systems
            for seed in seeds_by_system[system]
        })
        if len(seeds) <= 1:
            identical[dataset] = True
            continue
        references = []
        for seed in seeds:
            system = next(
                system for system in systems if seed in seeds_by_system[system]
            )
            references.append(_item_signature(frames[(system, dataset, seed)]))
        identical[dataset] = all(
            references[0].equals(reference) for reference in references[1:]
        )
    return identical


def _load(systems, seeds_by_system, files) -> BenchmarkData:
    frames = {}
    scores = {}
    thresholds = {}
    for path in files:
        match = PRED_RE.fullmatch(path.name)
        if not match:
            raise ValueError(f"nom de fichier de scores inattendu : {path.name}")
        system, dataset, seed = match.group(1), match.group(2), int(match.group(3))
        frame = pd.read_csv(path).reset_index(drop=True)
        _validate_frame(frame, path, dataset)
        frames[(system, dataset, seed)] = frame
        scores[(system, dataset, seed)] = frame["score"].to_numpy(dtype=float)
        thresholds[(system, dataset, seed)] = float(frame["threshold"].iloc[0])

    _validate_cross_system_items(frames, systems, seeds_by_system)
    identical_by_dataset = _identical_by_dataset(
        frames, systems, seeds_by_system
    )
    return BenchmarkData(
        frames=frames,
        scores=scores,
        thresholds=thresholds,
        systems=systems,
        seeds_by_system=seeds_by_system,
        files=tuple(files),
        identical_by_dataset=identical_by_dataset,
    )


def _groups(frame: pd.DataFrame):
    group_ids = frame["group_id"].to_numpy()
    universe = pd.unique(group_ids)
    groups = {
        group_id: np.flatnonzero(group_ids == group_id)
        for group_id in universe
    }
    return universe, groups


def _pair_indices(frame: pd.DataFrame) -> list[tuple[int, int]] | None:
    if "pair_id" not in frame.columns:
        return None
    pairs = []
    for _, group in frame.groupby("pair_id", sort=False):
        index_0 = group.index[group["label"] == 0].to_numpy()
        index_1 = group.index[group["label"] == 1].to_numpy()
        pairs.append((int(index_0[0]), int(index_1[0])))
    return pairs


def _pair_accuracy_selected(pairs, scores, counts) -> float:
    weighted_sum = 0.0
    total_weight = 0.0
    for index_0, index_1 in pairs:
        weight = float(counts[index_0])
        if weight == 0:
            continue
        difference = scores[index_1] - scores[index_0]
        value = 1.0 if difference > 0 else 0.5 if difference == 0 else 0.0
        weighted_sum += weight * value
        total_weight += weight
    return weighted_sum / total_weight if total_weight else np.nan


def _metrics_on_indices(data, system, dataset, seed, indices, counts, pairs):
    frame = data.frames[(system, dataset, seed)]
    labels = frame["label"].to_numpy(dtype=int)[indices]
    if len(np.unique(labels)) < 2:
        return None
    scores = data.scores[(system, dataset, seed)][indices]
    threshold = data.thresholds[(system, dataset, seed)]
    values = {
        "auroc": float(_auroc(labels, scores)),
        "macro_f1": float(_macro_f1(labels, scores >= threshold)),
    }
    if pairs is not None:
        values["pair_acc"] = _pair_accuracy_selected(
            pairs, data.scores[(system, dataset, seed)], counts
        )
    return values


def evaluate(data: BenchmarkData, n_boot: int, seed: int):
    """Compute metrics and paired differences from loaded score files."""
    metric_records = []
    paired_records = []

    for dataset in DATASETS:
        metrics = _metrics_for_dataset(dataset)
        paired_metrics = ("auroc",) + (
            ("pair_acc",) if dataset in PAIR_DATASETS else ()
        )
        all_seeds = sorted({
            eval_seed
            for system in data.systems
            for eval_seed in data.seeds_by_system[system]
        })
        reference_system = {
            eval_seed: next(
                system for system in data.systems
                if eval_seed in data.seeds_by_system[system]
            )
            for eval_seed in all_seeds
        }
        universes = {}
        groups = {}
        pairs_by_seed = {}
        for eval_seed in all_seeds:
            frame = data.frames[
                (reference_system[eval_seed], dataset, eval_seed)
            ]
            universes[eval_seed], groups[eval_seed] = _groups(frame)
            pairs_by_seed[eval_seed] = _pair_indices(frame)

        point = {}
        for system in data.systems:
            for eval_seed in data.seeds_by_system[system]:
                point[(system, eval_seed)] = compute_metrics(
                    data.frames[(system, dataset, eval_seed)],
                    data.scores[(system, dataset, eval_seed)],
                    data.thresholds[(system, dataset, eval_seed)],
                )

        boot = {}
        rejected = {}
        rejected_aggregate = {}
        paired_boot = {}
        paired_rejected = {}

        def add_boot(key, value):
            boot.setdefault(key, []).append(value)

        universe = np.unique(np.concatenate([
            universes[eval_seed] for eval_seed in all_seeds
        ]))
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            drawn = bootstrap_prompt_ids(universe, rng)
            current = {}
            for eval_seed in all_seeds:
                frame = data.frames[
                    (reference_system[eval_seed], dataset, eval_seed)
                ]
                indices = draw_cluster_indices(drawn, groups[eval_seed])
                counts = np.bincount(indices, minlength=len(frame))
                labels = frame["label"].to_numpy(dtype=int)[indices]
                if len(np.unique(labels)) < 2:
                    for system in data.systems:
                        if eval_seed in data.seeds_by_system[system]:
                            rejected[(system, eval_seed)] = (
                                rejected.get((system, eval_seed), 0) + 1
                            )
                    continue
                for system in data.systems:
                    if eval_seed not in data.seeds_by_system[system]:
                        continue
                    values = _metrics_on_indices(
                        data, system, dataset, eval_seed,
                        indices, counts, pairs_by_seed[eval_seed],
                    )
                    current[(system, eval_seed)] = values
                    for metric in metrics:
                        add_boot(
                            (system, eval_seed, metric),
                            values[metric],
                        )

            for system in data.systems:
                for metric in metrics:
                    aggregate = _nanmean([
                        current[(system, eval_seed)][metric]
                        for eval_seed in data.seeds_by_system[system]
                        if (system, eval_seed) in current
                    ])
                    if np.isfinite(aggregate):
                        add_boot((system, "mean", metric), aggregate)
                    else:
                        rejected_aggregate[(system, metric)] = (
                            rejected_aggregate.get((system, metric), 0) + 1
                        )

            for pair in PAIRS:
                for metric in paired_metrics:
                    diffs = []
                    for eval_seed in data.seeds_by_system[pair[0]]:
                        if (
                            (pair[0], eval_seed) in current
                            and (pair[1], eval_seed) in current
                        ):
                            diffs.append(
                                current[(pair[0], eval_seed)][metric]
                                - current[(pair[1], eval_seed)][metric]
                            )
                    difference = _nanmean(diffs)
                    if np.isfinite(difference):
                        paired_boot.setdefault((pair, metric), []).append(
                            difference
                        )
                    else:
                        paired_rejected[(pair, metric)] = (
                            paired_rejected.get((pair, metric), 0) + 1
                        )

        for system in data.systems:
            seeds = data.seeds_by_system[system]
            for eval_seed in seeds:
                frame = data.frames[(system, dataset, eval_seed)]
                for metric in metrics:
                    estimate = point[(system, eval_seed)][metric]
                    ci_low, ci_high = _interval(
                        boot.get((system, eval_seed, metric), [])
                    )
                    metric_records.append({
                        "system": system,
                        "dataset": dataset,
                        "seed": eval_seed,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "sd_across_seeds": np.nan,
                        "n": len(frame),
                        "n_clusters": int(frame["group_id"].nunique()),
                        "n_seeds": 1,
                        "n_boot": n_boot,
                        "n_boot_rejected": rejected.get(
                            (system, eval_seed), 0
                        ),
                    })
            if len(seeds) > 1:
                for metric in metrics:
                    estimates = [
                        point[(system, eval_seed)][metric]
                        for eval_seed in seeds
                    ]
                    finite = [
                        float(value) for value in estimates
                        if np.isfinite(value)
                    ]
                    estimate = _nanmean(estimates)
                    sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else np.nan
                    ci_low, ci_high = _interval(
                        boot.get((system, "mean", metric), [])
                    )
                    metric_records.append({
                        "system": system,
                        "dataset": dataset,
                        "seed": "mean",
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "sd_across_seeds": sd,
                        "n": float(np.mean([
                            len(data.frames[(system, dataset, eval_seed)])
                            for eval_seed in seeds
                        ])),
                        "n_clusters": float(np.mean([
                            int(data.frames[
                                (system, dataset, eval_seed)
                            ]["group_id"].nunique())
                            for eval_seed in seeds
                        ])),
                        "n_seeds": len(seeds),
                        "n_boot": n_boot,
                        "n_boot_rejected": rejected_aggregate.get(
                            (system, metric), 0
                        ),
                    })

        p_seeds = data.seeds_by_system[PAIR_SYSTEMS[0]]
        for pair in PAIRS:
            for metric in paired_metrics:
                point_diffs = [
                    point[(pair[0], eval_seed)][metric]
                    - point[(pair[1], eval_seed)][metric]
                    for eval_seed in p_seeds
                ]
                difference = _nanmean(point_diffs)
                values = paired_boot.get((pair, metric), [])
                ci_low, ci_high = _interval(values)
                paired_records.append({
                    "comparison": f"{pair[0]}-{pair[1]}",
                    "dataset": dataset,
                    "metric": metric,
                    "difference": difference,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p": float(_bootstrap_p_value(values)),
                    "p_holm": np.nan,
                    "n_seeds": len(p_seeds),
                    "n_boot": n_boot,
                    "n_boot_rejected": paired_rejected.get((pair, metric), 0),
                })

    auroc_indices = [
        index for index, record in enumerate(paired_records)
        if record["metric"] == "auroc"
    ]
    adjusted = holm([paired_records[index]["p"] for index in auroc_indices])
    for index, value in zip(auroc_indices, adjusted):
        paired_records[index]["p_holm"] = float(value)

    metric_columns = [
        "system", "dataset", "seed", "metric", "estimate", "ci_low",
        "ci_high", "sd_across_seeds", "n", "n_clusters", "n_seeds",
        "n_boot", "n_boot_rejected",
    ]
    paired_columns = [
        "comparison", "dataset", "metric", "difference", "ci_low",
        "ci_high", "p", "p_holm", "n_seeds", "n_boot",
        "n_boot_rejected",
    ]
    metrics_df = pd.DataFrame(metric_records, columns=metric_columns)
    metrics_df = metrics_df.sort_values(
        ["dataset", "system", "metric", "seed"], kind="stable"
    ).reset_index(drop=True)
    paired_df = pd.DataFrame(paired_records, columns=paired_columns)
    paired_df = paired_df.sort_values(
        ["dataset", "comparison", "metric"], kind="stable"
    ).reset_index(drop=True)
    return metrics_df, paired_df


def run(
    preds_dir,
    out_dir,
    n_boot=2000,
    seed=13,
    expect_seeds=DEFAULT_EXPECT_SEEDS,
    overwrite=False,
):
    preds_dir = Path(preds_dir)
    out_dir = Path(out_dir)
    if n_boot < 1:
        raise ValueError("--boot doit être strictement positif")

    now = datetime.now()
    info_path = out_dir / (
        f"benchmark_eval_INFO_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"
    )
    metrics_path = out_dir / "benchmark_metrics.csv"
    paired_path = out_dir / "benchmark_paired.csv"
    if not overwrite:
        existing = [
            path for path in (metrics_path, paired_path, info_path)
            if path.exists()
        ]
        if existing:
            listed = ", ".join(str(path) for path in existing)
            raise FileExistsError(
                f"fichier(s) déjà présent(s), utilisez --overwrite : {listed}"
            )

    systems, seeds_by_system, files = discover_score_files(
        preds_dir, expect_seeds
    )
    data = _load(systems, seeds_by_system, files)
    metrics_df, paired_df = evaluate(data, n_boot, seed)

    info = {
        "date": now.isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "seed": seed,
        "n_boot": n_boot,
        "git_commit": git_commit(),
        "systems": list(systems),
        "seeds_by_system": {
            system: list(seeds) for system, seeds in seeds_by_system.items()
        },
        "datasets": list(DATASETS),
        "pair_systems": list(PAIR_SYSTEMS),
        "pairs": [f"{left}-{right}" for left, right in PAIRS],
        "inputs": {path.name: _sha256(path) for path in files},
        "n_inputs": len(files),
        "identical_items_across_seeds": data.identical_by_dataset,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    paired_df.to_csv(paired_path, index=False)
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    return metrics_df, paired_df, info


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark score files (secondary analysis)."
    )
    parser.add_argument(
        "--preds", type=Path, default=ROOT / "results" / "preds"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--expect-seeds", type=int, nargs="+",
        default=list(DEFAULT_EXPECT_SEEDS),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        args.preds,
        args.out,
        n_boot=args.boot,
        seed=args.seed,
        expect_seeds=args.expect_seeds,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
