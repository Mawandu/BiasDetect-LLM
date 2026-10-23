# Revue de Code — Tâche 3 (`agent/audit..HEAD`)

**Statut : OK**

## 1. Vérification du diff (`git diff agent/audit..HEAD`)
- **`biasdetect/models/systems.py:183`** : `self.fz = SymbolicFeaturizer()` a été remplacé par `self.fz = SymbolicFeaturizer(seed=self.seed)` dans `SymbolicLR.fit`, aligné sur `HybridFusion.fit` (`biasdetect/models/systems.py:239`).
- **`tests/test_biasdetect.py:1-63`** : ajout de `DummyFeaturizer` (`tests/test_biasdetect.py:14-29`) et de deux tests unitaires sans accès réseau ni chargement de modèle :
  - `test_symbolic_lr_passes_seed_to_featurizer` (`tests/test_biasdetect.py:41-51`) : instancie `SymbolicLR()`, fixe `sysobj.seed = 21`, appelle `sysobj.fit(train, dev)` avec `SymbolicFeaturizer` remplacé par `DummyFeaturizer` via `monkeypatch`, et vérifie `assert DummyFeaturizer.calls == [21]`.
  - `test_symbolic_lr_default_seed_is_13` (`tests/test_biasdetect.py:54-63`) : vérifie que la seed par défaut (`13`) est bien transmise.
- **`docs/agents/TASKS.md:21`** : la Tâche 3 est marquée `[x]`.

## 2. Vérification du critère de réussite de la Tâche 3
- **Transmission de la seed** : `SymbolicLR.fit` (`biasdetect/models/systems.py:183`) passe explicitement `seed=self.seed` à `SymbolicFeaturizer`.
- **Échec du test sans le correctif** : sans `seed=self.seed` à `biasdetect/models/systems.py:183`, `DummyFeaturizer.__init__(self, ontology=None, seed=13)` (`tests/test_biasdetect.py:20-22`) reçoit la valeur par défaut `13` et enregistre `DummyFeaturizer.calls == [13]`, ce qui fait échouer `assert DummyFeaturizer.calls == [21]` (`tests/test_biasdetect.py:51`).

## 3. Résultat des tests (`python -m pytest -q`)
```text
........                                                                 [100%]
8 passed in 27.35s
```

Aucun problème détecté. Aucune tâche supplémentaire ajoutée dans `docs/agents/TASKS.md`.

---

# Revue de Code — Tâche 2 (`61ec55a..ca3f4e6`)

**Statut : OK**

## 1. Vérification du diff (`git diff 61ec55a..ca3f4e6`)
- **`requirements.txt:1-13`** :
  - Suppression des paquets inutilisés `plotly` et `matplotlib` (aucun import dans le dépôt).
  - Ajout de `scikit-learn`, `scipy`, `rdflib`, `PyYAML`, `openai`, `datasets`, `tqdm`.
- **`requirements-train.txt:1-2`** :
  - Création du fichier dédiant `torch` et `transformers` à l'entraînement / inférence lourde (`B3`/`B4`).
- **`docs/agents/TASKS.md:16`** : la Tâche 2 est marquée `[x]` (et la Tâche 9 à `docs/agents/TASKS.md:63-66` prévoit déjà la documentation de `requirements-train.txt` et le déplacement de `pytest` vers `requirements-dev.txt`).

## 2. Couverture exhaustive des imports tiers (`biasdetect/`, `modules/`, `experiments/`, `main.py`)
Tous les imports tiers (au niveau module comme à l'intérieur des fonctions/méthodes) sont couverts par `requirements.txt` et `requirements-train.txt`, sans paquet inutile :
- `streamlit>=1.30.0` (`requirements.txt:1`) : `main.py:1`
- `spacy>=3.7.0` (`requirements.txt:2`) : `modules/symbolic.py:1-2`
- `sentence-transformers>=2.3.0` (`requirements.txt:3`) : `modules/linguistic.py:1`, `biasdetect/features/neural.py:26` (dans `_get_model`)
- `numpy` (`requirements.txt:4`) : `modules/linguistic.py:2`, `biasdetect/data/loaders.py:16`, `biasdetect/eval/metrics.py:4`, `biasdetect/features/neural.py:7`, `biasdetect/features/symbolic.py:8`, `biasdetect/models/systems.py:11`, `experiments/generate_bold.py:13`, `experiments/run.py:23`, `experiments/train_transformer.py:14`
- `pandas` (`requirements.txt:5`) : `biasdetect/annotation/agreement.py:15`, `biasdetect/annotation/export.py:10`, `biasdetect/data/loaders.py:17`, `biasdetect/eval/metrics.py:5`, `biasdetect/models/systems.py:12`, `biasdetect/ontology/build.py:26`, `experiments/generate_bold.py:14`, `experiments/run.py:24`
- `scikit-learn` (`requirements.txt:6`) : `biasdetect/annotation/agreement.py:16`, `biasdetect/eval/metrics.py:7`, `biasdetect/models/systems.py:13-16`
- `scipy` (`requirements.txt:7`) : `biasdetect/eval/metrics.py:6`
- `rdflib` (`requirements.txt:8`) : `biasdetect/features/symbolic.py:9-10`, `biasdetect/ontology/build.py:27-28`
- `PyYAML` (`requirements.txt:9`) : `biasdetect/llm/client.py:17`
- `openai` (`requirements.txt:10`) : `biasdetect/llm/client.py:59` (dans `LLMClient.__post_init__`)
- `datasets` (`requirements.txt:11`) : `biasdetect/data/loaders.py:49` (dans `load_stereoset`), `experiments/generate_bold.py:26` (dans `sample_prompts`)
- `tqdm` (`requirements.txt:12`) : `biasdetect/models/systems.py:164` (dans `LLMJudge.score`), `experiments/generate_bold.py:15`
- `pytest` (`requirements.txt:13`) : `tests/test_modules.py:4`, `tests/test_biasdetect.py:7` (séparation dans `requirements-dev.txt` déjà planifiée en Tâche 9, `docs/agents/TASKS.md:63-66`)
- `torch` (`requirements-train.txt:1`) : `biasdetect/models/systems.py:86` (dans `ZeroShotNLI.fit`), `biasdetect/models/systems.py:117` (dans `FineTuned.fit`), `experiments/train_transformer.py:15-16`
- `transformers` (`requirements-train.txt:2`) : `biasdetect/models/systems.py:87` (dans `ZeroShotNLI.fit`), `biasdetect/models/systems.py:118` (dans `FineTuned.fit`), `experiments/train_transformer.py:17`

Aucun problème détecté.

---

# Revue de Code — Tâche 5 (`05b9b9c..408b452`)

**Statut : OK**

## 1. Vérification du diff (`git diff 05b9b9c..408b452`)
- **`tests/test_modules.py:28-34` (fixture `sym`)** : suppression de l'appel redondant `spacy.load("en_core_web_sm")`. La fixture instancie une seule fois `sym_analyzer = SymbolicAnalyzer()` (qui charge `en_core_web_sm` dans `modules/symbolic.py:11`) et appelle `pytest.skip("en_core_web_sm not installed")` si `sym_analyzer.nlp is None`. Le temps de setup de `test_symbolic_single_word` passe de `16.10s` (lors de l'audit initial) à `5.13s`.
- **`tests/test_modules.py:61-90` (`FakeSentenceTransformer` et fixture `fake_encoder`)** : `monkeypatch.setattr(linguistic, "SentenceTransformer", FakeSentenceTransformer)` (`tests/test_modules.py:89`) remplace `SentenceTransformer` par un encodeur déterministe 2D sans chargement de poids ni appel réseau.
- **`tests/test_modules.py:93-125`** : ajout de 5 tests unitaires couvrant `modules/linguistic.py` :
  - `test_linguistic_uses_mocked_model` (`tests/test_modules.py:93-96`)
  - `test_linguistic_empty_text` (`tests/test_modules.py:99-104`)
  - `test_linguistic_male_bias` (`tests/test_modules.py:107-111`)
  - `test_linguistic_female_bias` (`tests/test_modules.py:114-118`)
  - `test_linguistic_neutral_text` (`tests/test_modules.py:121-125`)
- **`docs/agents/TASKS.md:33`** : la Tâche 5 est marquée `[x]` (et la Tâche 10 à `docs/agents/TASKS.md:68-71` est déjà prévue pour tester un `raw_diff` intermédiaire de `0.05` et `0.1`).

## 2. Résultat des tests (`python -m pytest -q --durations=5`)
```text
.............                                                            [100%]
============================= slowest 5 durations ==============================
5.13s setup    tests/test_modules.py::test_symbolic_single_word
0.15s call     tests/test_biasdetect.py::test_symbolic_lr_passes_seed_to_featurizer
0.05s call     tests/test_biasdetect.py::test_symbolic_lr_default_seed_is_13
0.04s call     tests/test_modules.py::test_symbolic_single_word
0.03s call     tests/test_modules.py::test_symbolic_multi_word
13 passed in 47.25s
```

Aucun problème détecté.

---

# Revue de Code — Tâche 10 (`f0e8119..c70ff2d`)

**Statut : OK**

## 1. Vérification du diff (`git diff f0e8119..c70ff2d`)
- **`modules/linguistic.py`** : non modifié.
- **`tests/test_modules.py:129-168`** :
  - Ajout de `ControlledSentenceTransformer` (`tests/test_modules.py:129-143`), de `_unit_vector_with_diff(diff)` (`tests/test_modules.py:146-148`, qui construit un vecteur unitaire 2D `[x, x - diff]` de sorte que `male_sim - female_sim == diff`) et de la fixture `controlled_analyzer` (`tests/test_modules.py:151-160`).
  - Ajout du test paramétré `test_linguistic_intermediate_score` (`tests/test_modules.py:163-168`) vérifiant `raw_diff` (`0.05` et `0.1`), `score` avec `pytest.approx` sur les valeurs numériques attendues (`46.21` et `76.16`, sans recopier la formule `100 * np.tanh(abs(raw_diff) * 10)` de `modules/linguistic.py:50`) et `direction == "Masculin"`.
  - Si `* 10` était remplacé par `* 20` dans `modules/linguistic.py:50`, les scores vaudraient `76.16` (au lieu de `46.21`) et `96.40` (au lieu de `76.16`), faisant échouer `assert res["score"] == pytest.approx(expected_score, abs=0.1)` (`tests/test_modules.py:167`).
- **`docs/agents/TASKS.md:68`** : la Tâche 10 est marquée `[x]`.

## 2. Résultat des tests (`python -m pytest -q`)
```text
...............                                                          [100%]
15 passed in 50.61s
```

Aucun problème détecté.
