# BiasDetect-LLM

Corpus, annotation, detectors and evaluation code for **stereotype detection in
LLM-generated text**.

The project asks whether detectors of social stereotypes trained on benchmark
sentences transfer to text written by large language models, and whether a hybrid
detector that combines ontology-based symbolic features with a neural encoder
improves on its two components. It accompanies the article

> *From Benchmark Sentences to Generated Text: A Pre-Specified Evaluation of
> Symbolic, Neural and Hybrid Stereotype Detectors.*

Main findings (details and caveats in the article and in `docs/analysis_plan.md`):

- On the primary analysis (282 texts, 17 prompts, no refusals) no trained detector
  separates stereotypical from non-stereotypical LLM texts above chance
  (AUROC 0.20 to 0.44).
- On benchmarks the hybrid has a small advantage on StereoSet (AUROC 0.700), the
  symbolic detector is slightly better on CrowS-Pairs, and all systems are near
  chance on SBIC.

## Repository layout

| Path | Content |
| --- | --- |
| `corpus/` | **BiasDetect-Corpus**: 432 LLM-generated English texts with three annotators' labels (`biasdetect_corpus.csv`), the label codebook and the agreement summary (`README.md`), and SHA-256 hashes (`SHA256SUMS`). |
| `corpus/generation/` | Script, prompts and raw outputs used to generate the texts (see its `README.md`). API keys are **not** included. |
| `docs/analysis_plan.md` | The analysis plan, committed before any detector was scored, with the later precisions and the post hoc exploratory analyses clearly separated. |
| `docs/annotation_guide.md` | The written guide given to the annotators. |
| `biasdetect/` | Library code: ontology construction (`ontology/`), features (`features/`), detectors (`models/systems.py`), data loaders (`data/`), metrics (`eval/`), agreement statistics (`annotation/`). |
| `experiments/` | Scripts that score the corpus (`score_corpus.py`), evaluate it (`eval_corpus.py`) and evaluate the benchmarks (`eval_benchmarks.py`), plus the benchmark runner (`run.py`). |
| `scripts/build_corpus.py` | Builds the corpus file from the annotation exports. |
| `tests/` | Unit tests (no network, no model download). |
| `configs/` | Configuration of the optional LLM-judge baseline. |
| `modules/`, `main.py`, `architecture.md` | Code of the earlier prototype, kept for reference. It is not used by the experiments of the article. |

## Systems compared

| Name | Description |
| --- | --- |
| B0 | Majority class (AUROC 0.5 by construction). |
| B1 | Legacy keyword-rule detector. |
| B2 | Legacy detector based on embedding similarity. |
| `P_sym` | **Symbolic**: logistic regression on features computed from the ontology. |
| `P_neu_minilm` | **Neural**: logistic regression on frozen `all-MiniLM-L6-v2` embeddings. |
| `P_fusion_minilm` | **Hybrid**: stacked late fusion of the neural score and the symbolic features. |

All trained systems learn from the StereoSet training split only. The decision
threshold is chosen on the StereoSet development split and never on the corpus.
The ontology attributes used for training are cross-fitted by target group to avoid
leakage. Five seeds are used (13, 21, 42, 7, 99).

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt     # for the tests
```

The first run downloads the `sentence-transformers` encoder and the benchmark
datasets (StereoSet, CrowS-Pairs, SBIC) through `biasdetect/data/loaders.py`.

## Running the tests

```bash
pytest -q
```

The tests use fake encoders and synthetic data; they need no network access.

## Reproducing the results

The scripts refuse to overwrite existing result files and record the hashes of
their inputs in a run-information file next to their outputs. Run each script with
`--help` for its options.

```bash
python -m experiments.score_corpus --help    # scores the 432 texts (60 score files)
python -m experiments.eval_corpus --help     # primary, sensitivity and exploratory corpus analyses
python -m experiments.run --help             # trains and evaluates on the benchmarks
python -m experiments.eval_benchmarks --help # secondary analysis on the benchmarks
```

Order: benchmarks (`run`), corpus scoring (`score_corpus`), then the two evaluation
scripts. Results are written to `results/`, which is not versioned; the figures of
the article come from the runs dated 29 October 2026 (see the run-information files).

## The corpus

- 432 texts = 24 prompts × 6 models × 3 samples per prompt; temperature 0.8.
- Models: gpt-4o-mini, claude-sonnet-4-6, gpt-oss-120b, gpt-oss-20b, qwen3.8-27b and
  open-mistral-nemo.
- Categories: GENDER, ORIGIN, RELIGION (core) and OTHER.
- Labels: `1` = stereotype present, `0` = not stereotyped, `?` = undecidable;
  the primary gold label is the majority vote of three annotators. The codebook
  is in `corpus/README.md`.
- Integrity: check the file against `corpus/SHA256SUMS` (`sha256sum -c`).

**Ethics and content.** The corpus contains stereotypical statements about social
groups generated by language models. It is released for research on detection and
mitigation only. Annotator identities are not part of the release.

## Citation

If you use this repository, please cite the article and the archived release:

```
BiasDetect-LLM, version 1.0. Zenodo. https://doi.org/10.5281/zenodo.23803926
```

A machine-readable `CITATION.cff` is added with the release.
