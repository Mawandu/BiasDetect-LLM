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
