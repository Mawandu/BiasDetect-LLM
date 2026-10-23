# Tâches de Correction — BiasDetect-LLM

Chaque tâche doit être réalisée **une à la fois** sur la branche `agent/fix`, suivie de `.venv/bin/python -m pytest -q`, d'un commit, et du marquage `[x]`.
Rappel (`AGENTS.md`) : `data/` et `results/` sont en lecture seule ; aucun appel réseau ni téléchargement dans les tests (utiliser des mocks) ; ne jamais lire/afficher/modifier `.env`.

---

- [x] **Tâche 1 : Corriger `.gitignore` pour ne plus ignorer `biasdetect/data/` et ajouter `biasdetect/annotation/__init__.py`**
  - **Fichiers** : `.gitignore`, `biasdetect/data/__init__.py`, `biasdetect/data/loaders.py`, `biasdetect/annotation/__init__.py`
  - **Action** :
    1. Dans `.gitignore:19-20`, remplacer `data/` par `/data/` et `results/` par `/results/` afin que seul le dossier `/data/` à la racine soit ignoré et que `biasdetect/data/` soit suivi par Git.
    2. Créer un fichier vide `biasdetect/annotation/__init__.py`.
    3. Ajouter `biasdetect/data/__init__.py`, `biasdetect/data/loaders.py` et `biasdetect/annotation/__init__.py` au suivi Git.
  - **Critère de réussite vérifiable** : `git check-ignore -v biasdetect/data/loaders.py` ne retourne rien (code retour `1`) et `git ls-files biasdetect/data/loaders.py biasdetect/annotation/__init__.py` affiche bien les deux fichiers.

- [x] **Tâche 2 : Mettre à jour `requirements.txt` avec les dépendances réelles du projet**
  - **Fichiers** : `requirements.txt`
  - **Action** : Ajouter les bibliothèques utilisées par `biasdetect/` et `experiments/` (`rdflib`, `scikit-learn`, `scipy`, `PyYAML`, `openai`, `datasets`, `tqdm`, `torch`, `transformers`) dans `requirements.txt`.
  - **Critère de réussite vérifiable** : Toutes les bibliothèques tierces importées dans `biasdetect/`, `modules/`, `experiments/` et `main.py` figurent dans `requirements.txt`, et `.venv/bin/python -m pytest -q` passe.

- [x] **Tâche 3 : Corriger la fuite de données (seed non transmise) dans `SymbolicLR.fit`**
  - **Fichiers** : `biasdetect/models/systems.py`
  - **Action** : Dans `SymbolicLR.fit` (`biasdetect/models/systems.py:182-186`), remplacer `self.fz = SymbolicFeaturizer()` par `self.fz = SymbolicFeaturizer(seed=self.seed)` (comme dans `HybridFusion.fit` à la ligne 239).
  - **Critère de réussite vérifiable** : Après appel à `sys = SymbolicLR(); sys.seed = 21` (avec `SymbolicFeaturizer` mocké), `SymbolicFeaturizer` est bien appelé avec `seed=21`.

- [x] **Tâche 4 : Corriger `biasdetect/annotation/agreement.py` (création du dossier parent et cohérence docstring) et rendre `export.py` / `agreement.py` testables**
  - **Fichiers** : `biasdetect/annotation/agreement.py`, `biasdetect/annotation/export.py`
  - **Action** :
    1. Dans `biasdetect/annotation/agreement.py:8`, aligner la docstring sur le chemin réel `data/bold/gold.csv`.
    2. Dans `biasdetect/annotation/agreement.py:64`, s'assurer que le dossier parent existe avant l'écriture (`out_path = ROOT / "data" / "bold" / "gold.csv"; out_path.parent.mkdir(parents=True, exist_ok=True)`), ou permettre de paramétrer les chemins pour les tests unitaires sans toucher à `/data/`.
  - **Critère de réussite vérifiable** : Les fonctions de `export.py` et `agreement.py` s'exécutent sans erreur sur un répertoire temporaire (`tmp_path` pytest) sans modifier `/data/`.

- [x] **Tâche 5 : Optimiser la fixture `sym` dans `tests/test_modules.py` et tester `modules/linguistic.py` avec mock**
  - **Fichiers** : `tests/test_modules.py`
  - **Action** :
    1. Dans `tests/test_modules.py:25-33`, éviter le double appel à `spacy.load("en_core_web_sm")` en instanciant directement `SymbolicAnalyzer()` et en faisant `if sym_analyzer.nlp is None: pytest.skip("en_core_web_sm not installed")`.
    2. Ajouter des tests unitaires pour `modules/linguistic.py` (`LinguisticAnalyzer`) en mockant `SentenceTransformer` pour ne charger aucun modèle lourd et ne faire aucun accès réseau (tester texte vide, biais masculin, biais féminin, texte neutre).
  - **Critère de réussite vérifiable** : `.venv/bin/python -m pytest -q` passe et couvre `modules/linguistic.py`, `modules/symbolic.py` et `modules/fusion.py`.

- [ ] **Tâche 6 : Ajouter des tests unitaires (sans réseau ni chargement de modèle) pour `biasdetect/eval/metrics.py` et `biasdetect/data/loaders.py`**
  - **Fichiers** : `tests/test_biasdetect.py` (nouveau fichier)
  - **Action** :
    1. Tester toutes les fonctions de `biasdetect/eval/metrics.py` (`best_threshold`, `ece`, `metrics`, `pairwise_accuracy`, `bootstrap_ci`, `mcnemar`) sur des tableaux synthétiques déterministes.
    2. Tester `group_split` et `_sbic_category` de `biasdetect/data/loaders.py` sur un `DataFrame` synthétique en vérifiant l'absence de fuite de `group_id` entre `train`, `dev` et `test` (`set(train.group_id).isdisjoint(set(test.group_id))`).
  - **Critère de réussite vérifiable** : `.venv/bin/python -m pytest -q tests/test_biasdetect.py` passe en moins de 2 secondes sans accès réseau ni écriture dans `data/`.

- [ ] **Tâche 7 : Ajouter des tests unitaires (avec mocks/fixtures locales) pour `biasdetect/ontology/build.py`, `biasdetect/features/symbolic.py`, `biasdetect/features/neural.py`, `biasdetect/models/systems.py`, `biasdetect/llm/client.py` et `biasdetect/annotation/`**
  - **Fichiers** : `tests/test_biasdetect.py`
  - **Action** :
    1. Tester `biasdetect/ontology/build.py` (`_uri`, `_hb_descriptors`, `stereo_attributes` avec `load_processed` mocké) et `biasdetect/features/symbolic.py` (`SymbolicFeaturizer.analyze` et `transform` sur une mini-ontologie en mémoire ou `tmp_path`).
    2. Tester `biasdetect/features/neural.py` (`embed` avec `_get_model` et `CACHE` redirigés vers `tmp_path` via `monkeypatch`).
    3. Tester `biasdetect/models/systems.py` (`Majority`, `SymbolicLR`, `NeuralProbe`, `HybridFusion.fit/score/explain`) sur de petites données synthétiques avec embeddings et ontologie mockés.
    4. Tester `biasdetect/llm/client.py` (`load_config`, `LLMClient.chat_json` avec le client OpenAI mocké et variable d'environnement factice via `monkeypatch.setenv`, sans lire `.env`) et `biasdetect/annotation/` (`export.py` et `agreement.py` sur `tmp_path`).
  - **Critère de réussite vérifiable** : Tous les modules de `biasdetect/` sont couverts par au moins un test unitaire dans `tests/test_biasdetect.py`, aucun test ne fait d'accès réseau ni ne charge de modèle Transformer, et `.venv/bin/python -m pytest -q` passe intégralement.

- [ ] **Tâche 8 : Aligner `architecture.md` et `README.md` avec l'implémentation réelle du projet**
  - **Fichiers** : `architecture.md`, `README.md`
  - **Action** :
    1. Mettre à jour `architecture.md` et `README.md` pour refléter fidèlement l'architecture du dépôt : distinguer le démonstrateur Streamlit (`main.py` + `modules/`) et le pipeline expérimental hybride (`biasdetect/` + `experiments/`).
    2. Retirer ou qualifier clairement les éléments non implémentés (analyse stylométrique, score de toxicité, surlignage token par token dans le texte) et documenter les composants réellement implémentés (ontologie RDF/Turtle `S1`–`S4` avec provenance `prov:wasDerivedFrom`, `SymbolicFeaturizer`, sonde logistique `NeuralProbe`, fusion par stacking `HybridFusion` avec `GroupKFold`, et évaluation sur StereoSet, CrowS-Pairs et SBIC).
  - **Critère de réussite vérifiable** : Aucune composante absente du code n'est présentée comme implémentée dans `architecture.md` ou `README.md`.

- [ ] **Tâche 9 : documenter requirements-train.txt et séparer pytest**
  - **Fichiers** : README.md, requirements.txt, requirements-dev.txt
  - **Action** : Ajouter au README une section expliquant requirements-train.txt (entraînement, Colab) ; déplacer pytest dans requirements-dev.txt. Aucun pip install.
  - **Critère de réussite vérifiable** : grep -n "requirements-train" README.md affiche au moins une ligne et pytest n apparait plus dans requirements.txt.

- [x] **Tâche 10 : tester le score de LinguisticAnalyzer sur un biais intermédiaire**
  - **Fichiers** : tests/test_modules.py
  - **Action** : Ajouter un test avec un faux SentenceTransformer dont les vecteurs donnent un raw_diff connu (par exemple 0.05 puis 0.1) et verifier le score avec pytest.approx (environ 46 et 76, selon le commentaire de modules/linguistic.py). Ne pas modifier modules/linguistic.py. Aucun reseau, aucun chargement de modele.
  - **Critere de reussite verifiable** : le test passe, et remplacer temporairement "* 10" par "* 20" dans modules/linguistic.py le fait echouer.
