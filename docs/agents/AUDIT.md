# Audit du Projet BiasDetect-LLM

*Rôle : Auditeur / Testeur (Antigravity) — Branche : `agent/audit`*

---

## 1. Ce qui est fait et fonctionne (avec preuves)

### 1.1 Suite de tests existante (`tests/test_modules.py`)
Les 6 tests unitaires portant sur le prototype initial (`modules/fusion.py` et `modules/symbolic.py`) passent intégralement.

**Preuve** — commande exécutée : `.venv/bin/python -m pytest -q --durations=5`
```text
......                                                                   [100%]
============================= slowest 5 durations ==============================
16.10s setup    tests/test_modules.py::test_symbolic_single_word
0.08s call     tests/test_modules.py::test_symbolic_plural
0.06s call     tests/test_modules.py::test_symbolic_multi_word
0.05s call     tests/test_modules.py::test_symbolic_single_word
0.02s teardown tests/test_modules.py::test_symbolic_single_word
6 passed in 16.49s
```

### 1.2 Prototype de démonstration (`main.py` et `modules/`)
- **`modules/fusion.py:1-51` (`FusionEngine`)** : calcule la moyenne pondérée $\alpha \cdot \text{neural} + (1-\alpha) \cdot \text{symbolic}$ (`modules/fusion.py:14`) et génère une explication textuelle (`modules/fusion.py:26-47`). Testé et fonctionnel (`tests/test_modules.py:13-20`).
- **`modules/symbolic.py:8-145` (`SymbolicAnalyzer`)** : détecte par `PhraseMatcher` spaCy (`LOWER` et `LEMMA`, `modules/symbolic.py:55-63`) 15 termes répartis en 3 dictionnaires (`gender_rules`, `cultural_rules`, `representation_rules`, `modules/symbolic.py:19-43`), gère les expressions multi-mots et les pluriels, et plafonne le score à 100 (`modules/symbolic.py:139`). Testé et fonctionnel (`tests/test_modules.py:40-57`).
- **`modules/linguistic.py:5-69` (`LinguisticAnalyzer`)** : projette un texte via `SentenceTransformer` (`all-MiniLM-L6-v2` par défaut) et calcule la différence de similarité cosinus moyenne avec deux listes de 7 mots masculins et féminins (`modules/linguistic.py:11-14, 39-50`).
- **`main.py:1-122`** : interface Streamlit assemblant `LinguisticAnalyzer`, `SymbolicAnalyzer` et `FusionEngine` (`main.py:8-10, 20-25, 78-87`). Fonctionnement interactif de l'interface navigateur : **non vérifié** (aucun serveur lancé durant l'audit).

### 1.3 Pipeline expérimental et ontologie (`biasdetect/` et `experiments/`)
Plusieurs artefacts sur disque (en lecture seule dans `data/` et `results/`) prouvent que la chaîne de préparation des données, de construction de l'ontologie RDF et d'évaluation rapide (seed 13, encodeur `minilm`) a déjà été exécutée avec succès :
- **Jeux de données prétraités (`biasdetect/data/loaders.py:156-173`)** :
  - `data/processed/stereoset_s13.csv` (4 212 exemples, en-tête `text,label,category,group_id,pair_id,target,source,split`)
  - `data/processed/crows.csv` (2 084 exemples, en-tête `text,label,category,group_id,pair_id,source`)
  - `data/processed/sbic.csv` (4 413 exemples, en-tête `text,label,category,group_id,source`)
- **Ontologie RDF/Turtle (`biasdetect/ontology/build.py:93-152`)** :
  - `data/ontology/bias_s13.ttl` (5 001 lignes) construit à partir des 4 sources `S1` (`male_word_file.txt`, `female_word_file.txt`), `S2` (`descriptors.json`), `S3` (`occupations-stats.tsv`) et `S4` (log-odds sur StereoSet `train` split seed 13).
- **Cache d'embeddings (`biasdetect/features/neural.py:33-43`)** :
  - 5 fichiers `data/cache/minilm_*.npy` présents sur disque.
- **Évaluation de 6 systèmes sur la seed 13 (`experiments/run.py:50-75, 100-107`)** :
  - `results/metrics.csv` (18 lignes de résultats + en-tête) et 18 fichiers de prédictions dans `results/preds/*__s13.csv` pour `B0_majority`, `B1_legacy_rules`, `B2_legacy_embsim`, `P_sym`, `P_neu_minilm`, et `P_fusion_minilm`.
  - Poids appris de la fusion sauvegardés dans `results/weights_P_fusion_minilm_s13.csv` (`Macro-F1` sur `stereoset_test` : `0.581`, `AUROC` : `0.638`, `pair_acc` : `0.657` ; `Macro-F1` sur `crows` : `0.587` contre `0.373` pour `B1_legacy_rules` et `0.470` pour `B2_legacy_embsim`).

---

## 2. Ce qui est incomplet ou ne fonctionne pas

1. **Corpus BOLD et campagne d'annotation humaine absents (`experiments/generate_bold.py`, `biasdetect/annotation/export.py`, `biasdetect/annotation/agreement.py`)** :
   - Le dossier `data/bold/` (`generations.csv`, `gold.csv`) et le dossier `annotation/` (`items_master.csv`, `to_annotate_A1.csv`, `to_annotate_A2.csv`, `disagreements.csv`) n'existent pas sur le disque.
   - Preuve (commande `find biasdetect modules experiments tests data results annotation -maxdepth 3`) : le dossier `annotation` est absent (code retour `1`) et `data/` ne contient que `data/raw`, `data/cache`, `data/processed`, `data/ontology`.
   - `biasdetect/annotation/export.py:13,23` lit `data/bold/generations.csv` et échoue avec `FileNotFoundError` tant que la génération BOLD n'a pas eu lieu.
   - `biasdetect/annotation/agreement.py:19,23` lit `annotation/to_annotate_A1.csv` et échoue avec `FileNotFoundError`.
2. **Modèles fine-tunés `B4` absents (`biasdetect/models/systems.py:109-135`, `experiments/train_transformer.py`)** :
   - Le dossier `results/finetuned/` n'existe pas. L'appel à `FineTuned.fit()` (`biasdetect/models/systems.py:120-121`) lève explicitement `FileNotFoundError(f"{self.dir} missing: run experiments/train_transformer.py first (Colab)")`.
3. **Évaluation expérimentale partielle dans `results/metrics.csv`** :
   - Seule la seed `13` est présente dans `results/metrics.csv:2-19` (les seeds `21, 42, 7, 99` documentées dans `experiments/run.py:12` n'ont pas été tournées).
   - Parmi les 4 encodeurs définis dans `biasdetect/features/neural.py:13-18` (`minilm`, `mpnet`, `e5`, `bge`), seul `minilm` a été évalué dans `results/metrics.csv`. Les systèmes `P_neu_mpnet`, `P_neu_e5`, `P_neu_bge`, `P_fusion_mpnet`, `P_fusion_e5`, `P_fusion_bge`, `P_fusion_minilm+mpnet+e5+bge` (`biasdetect/models/systems.py:272-274`), `B3_nli_zeroshot` (`biasdetect/models/systems.py:78`) et `B5_llm_*` (`biasdetect/models/systems.py:138`) ne figurent pas dans `results/metrics.csv`.
4. **Application Streamlit (`main.py`) non reliée à la nouvelle architecture (`biasdetect/`)** :
   - `main.py:8-10` importe uniquement les classes prototypes de `modules/` (`LinguisticAnalyzer`, `SymbolicAnalyzer`, `FusionEngine`) et n'expose ni l'ontologie RDF (`biasdetect/features/symbolic.py`), ni le modèle hybride appris (`HybridFusion` dans `biasdetect/models/systems.py:208-267`).
5. **Documentation agent vide** :
   - Avant cet audit, `docs/agents/AUDIT.md`, `docs/agents/TASKS.md` et `docs/agents/REVIEW.md` étaient vides (0 octet).

---

## 3. Anomalies détaillées (avec `fichier:ligne` et preuves)

### Anomalie A1 — `biasdetect/data/` ignoré par Git à cause de `.gitignore:19` (imports cassés sur tout clone propre)
- **Localisation** :
  - `.gitignore:19` contient `data/` (sans `/` initial), ce qui ignore récursivement tout répertoire nommé `data/`, y compris le sous-package Python `biasdetect/data/` (`biasdetect/data/__init__.py` et `biasdetect/data/loaders.py`).
  - Les modules suivis par Git qui importent `biasdetect.data.loaders` sont :
    - `biasdetect/ontology/build.py:30` : `from biasdetect.data.loaders import load_processed`
    - `experiments/run.py:26` : `from biasdetect.data.loaders import load_processed`
    - `experiments/train_transformer.py:19` : `from biasdetect.data.loaders import load_processed`
- **Preuve (sortie de commandes)** :
  - `git check-ignore -v biasdetect/data/loaders.py` :
    ```text
    .gitignore:19:data/	biasdetect/data/loaders.py
    ```
  - `git ls-files biasdetect/data` ne retourne aucun fichier. Sur tout clone du dépôt, importer `biasdetect.ontology.build`, `experiments.run` ou `experiments.train_transformer` échoue avec `ModuleNotFoundError: No module named 'biasdetect.data'`.

### Anomalie A2 — Fuite de données (data leakage) dans `SymbolicLR.fit` sur les seeds $\neq 13$
- **Localisation** : `biasdetect/models/systems.py:182-186` vs `biasdetect/models/systems.py:238-239` et `biasdetect/features/symbolic.py:72`.
- **Détail** :
  - `experiments/run.py:53` définit `sysobj.seed = seed` avant d'appeler `sysobj.fit(train, dev)`.
  - `HybridFusion.fit` (`biasdetect/models/systems.py:239`) passe correctement la seed à l'ontologie : `self.fz = SymbolicFeaturizer(seed=self.seed)`.
  - En revanche, `SymbolicLR.fit` (`biasdetect/models/systems.py:183`) appelle `self.fz = SymbolicFeaturizer()` sans argument `seed`, ce qui utilise la valeur par défaut `seed: int = 13` (`biasdetect/features/symbolic.py:72`).
  - Or l'ontologie `bias_s13.ttl` inclut la source `S4` (`StereoAttribute`, `biasdetect/ontology/build.py:73-90, 138`) calculée sur le split `train` de StereoSet pour la seed `13`. Pour toute autre seed (ex. `21, 42, 7, 99`), des cibles du split `test` appartenaient au split `train` de la seed `13`, créant une fuite de données directe.
- **Preuve (`biasdetect/models/systems.py:182-186`)** :
  ```python
  def fit(self, train, dev):
      self.fz = SymbolicFeaturizer()
      self.clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
  ```

### Anomalie A3 — Écrasement d'attributs RDF lorsque `S1` et `S2` partagent le même terme dans `biasdetect/ontology/build.py`
- **Localisation** : `biasdetect/ontology/build.py:57-58, 105-125` et `biasdetect/features/symbolic.py:44-52`.
- **Détail** :
  - `_uri("term", term)` (`biasdetect/ontology/build.py:57-58`) génère le même URI `bd:term_<mot>` qu'un terme provienne de `S1` (lignes 117-119, avec `pole="male"` ou `"female"`) ou de `S2` (HolisticBias, lignes 121-125, avec `pole=None` et `preference="reviewed"` ou `"dispreferred"`).
  - Lorsqu'un terme figure à la fois dans `S1` et dans un autre axe de `S2`, `add_term` ajoute plusieurs triplets `bd:category` et `prov:wasDerivedFrom` sur le même nœud RDF `u`.
  - À la lecture dans `Ontology.__init__` (`biasdetect/features/symbolic.py:46-51`), `g.value(s, BD.category)` et `g.value(s, PROV.wasDerivedFrom)` retournent un seul objet arbitraire parmi ceux attachés au sujet `s` sans garantie de cohérence entre `category`, `pole` et `source`.

### Anomalie A4 — Incohérences entre `architecture.md` / `README.md` et le code
- **A4.1 — Module d'Analyse Stylométrique fantôme** :
  - `architecture.md:16, 28` inclut `Stylometry[Analyse Stylométrique]` dans la Branche 1 et sa connexion à `Merger((Fusion Hybride))`.
  - **Preuve** : aucune classe, fonction ou variable de stylométrie n'existe dans `modules/` ni dans `biasdetect/`.
- **A4.2 — Tests WEAT/SEAT et score de Toxicité absents** :
  - `architecture.md:15, 61, 89-90` et `README.md:23` annoncent le calcul de tests d'associations **WEAT/SEAT** (cibles vs attributs) et de **Toxicité** (`Calculer scores linguistiques (WEAT, Toxicité)` à `architecture.md:61`).
  - **Preuve** : `modules/linguistic.py:11-14, 39-50` ne calcule que la différence de cosinus entre la phrase et deux listes fixes de 7 mots masculins/féminins (sans concepts cibles WEAT/SEAT ni modèle de toxicité). `biasdetect/features/neural.py:13-43` et `biasdetect/models/systems.py:192-206` utilisent une sonde logistique (`LogisticRegression`) sur embeddings de phrases gelés.
- **A4.3 — Reconnaissance d'entités nommées (NER) et ontologie dans la branche symbolique** :
  - `README.md:53` (« Nous extrayons les entités nommées (personnes, lieux, professions) avec Spacy et les confrontons à une base de connaissances (ontologie) ») et `architecture.md:20-23, 93` (Ontologies RDF/OWL après `EntityRec[Reconnaissance d'Entités]`).
  - **Preuve** :
    - Dans `modules/symbolic.py:18-43`, il n'y a pas d'ontologie RDF/OWL mais un dictionnaire Python en dur (`# Mock Ontology`), et les entités nommées spaCy (`len(doc.ents)` à `modules/symbolic.py:144`) ne sont jamais confrontées aux règles (seul `PhraseMatcher` est utilisé à `modules/symbolic.py:78-80`).
    - Dans `biasdetect/features/symbolic.py:15, 81`, l'ontologie RDF/Turtle est bien chargée avec `rdflib`, mais l'extraction de termes se fait par expression régulière (`TOKEN_RE = re.compile(r"[a-z][a-z'\-]+")`) et n-grammes (`biasdetect/features/symbolic.py:75-78`), sans NER spaCy.
- **A4.4 — Surlignage Token/Phrase et suggestions de mitigation** :
  - `architecture.md:37` (`Highlight[Surlignage Token/Phrase]`) et `README.md:44, 54` (« en surlignant les termes problématiques »).
  - **Preuve** : ni `modules/fusion.py:26-47`, ni `main.py:89-121`, ni `biasdetect/models/systems.py:257-267` ne surlignent les tokens ou phrases dans le texte original.

### Anomalie A5 — Duplication de code entre `modules/` et `biasdetect/`
- **Localisation** :
  - Analyse neuronale : `modules/linguistic.py:5-69` (`LinguisticAnalyzer`) vs `biasdetect/features/neural.py:1-43` + `biasdetect/models/systems.py:62-75` (`LegacyGenderSim`, qui recode en dur à `biasdetect/models/systems.py:68-69` les mêmes listes de 7 mots masculins et féminins que `modules/linguistic.py:12-13` au lieu d'appeler `LinguisticAnalyzer`).
  - Analyse symbolique : `modules/symbolic.py:8-145` (`SymbolicAnalyzer`) vs `biasdetect/features/symbolic.py:34-132` (`Ontology`, `SymbolicFeaturizer`).
  - Fusion et explicabilité : `modules/fusion.py:1-48` (`FusionEngine`) vs `biasdetect/models/systems.py:208-267` (`HybridFusion`).

### Anomalie A6 — `requirements.txt` incomplet et désynchronisé
- **Localisation** : `requirements.txt:1-8`.
- **Détail** :
  - Dépendances importées dans `biasdetect/` et `experiments/` mais **absentes** de `requirements.txt` :
    - `rdflib` (`biasdetect/ontology/build.py:27-28`, `biasdetect/features/symbolic.py:9-10`)
    - `scikit-learn` (`biasdetect/eval/metrics.py:7`, `biasdetect/models/systems.py:13-16`, `biasdetect/annotation/agreement.py:16`)
    - `scipy` (`biasdetect/eval/metrics.py:6`)
    - `PyYAML` (`biasdetect/llm/client.py:17`)
    - `openai` (`biasdetect/llm/client.py:59`)
    - `datasets` (`biasdetect/data/loaders.py:49`, `experiments/generate_bold.py:26`)
    - `tqdm` (`biasdetect/models/systems.py:164`, `experiments/generate_bold.py:15`)
    - `torch` et `transformers` (`biasdetect/models/systems.py:86-87, 117-118`, `experiments/train_transformer.py:15-17`)
  - Dépendances listées dans `requirements.txt:6-7` mais **jamais importées** dans le projet : `plotly`, `matplotlib`.

### Anomalie A7 — Incohérence de chemin et absence de `mkdir` dans `biasdetect/annotation/agreement.py` et absence de `__init__.py`
- **Localisation** : `biasdetect/annotation/agreement.py:8, 64` et dossier `biasdetect/annotation/`.
- **Détail** :
  - `biasdetect/annotation/` est le seul sous-dossier de `biasdetect/` sans fichier `__init__.py`.
  - La docstring `biasdetect/annotation/agreement.py:8` documente la sortie `annotation/gold.csv` (avec mention `(data/bold/gold.csv)`), et la ligne 64 exécute `out.to_csv(ROOT / "data" / "bold" / "gold.csv")` sans créer le dossier parent `(ROOT / "data" / "bold").mkdir(parents=True, exist_ok=True)`, ce qui lève `OSError` si `data/bold/` n'existe pas.

### Anomalie A8 — Double chargement du modèle spaCy dans `tests/test_modules.py`
- **Localisation** : `tests/test_modules.py:25-33` et `modules/symbolic.py:11`.
- **Détail** : La fixture `sym` charge `spacy.load("en_core_web_sm")` à `tests/test_modules.py:29` uniquement pour vérifier sa présence (en jetant l'objet retourné), puis instancie `SymbolicAnalyzer()` à `tests/test_modules.py:33`, qui rappelle `spacy.load("en_core_web_sm")` à `modules/symbolic.py:11`.
- **Preuve** : `16.10s setup tests/test_modules.py::test_symbolic_single_word` mesuré par `python -m pytest -q --durations=5`.

---

## 4. Couverture de `biasdetect/` par les tests et analyse des accès réseau / chargements de modèles

### 4.1 Couverture par module de `biasdetect/`
Le seul fichier de test du dépôt est `tests/test_modules.py:1-57`, qui teste uniquement `modules/fusion.py` (`test_fusion_weighted_average` à `tests/test_modules.py:13`, `test_fusion_no_bias_message` à `tests/test_modules.py:18`) et `modules/symbolic.py` (`test_symbolic_single_word` à `tests/test_modules.py:40`, `test_symbolic_multi_word` à `tests/test_modules.py:44`, `test_symbolic_plural` à `tests/test_modules.py:52`, `test_symbolic_empty` à `tests/test_modules.py:56`). Note : `modules/linguistic.py` n'est couvert par aucun test.

Voici l'état pour chaque module de `biasdetect/` :

| Module de `biasdetect/` | Couvert par au moins un test ? | Preuve (nom du test) | Fonctions / classes non testées |
| :--- | :--- | :--- | :--- |
| `biasdetect/__init__.py` | **Non** | Aucun test dans `tests/test_modules.py` | (Fichier vide) |
| `biasdetect/data/loaders.py` | **Non** | Aucun test dans `tests/test_modules.py` | `_download` (`:31`), `load_stereoset` (`:42`), `load_crows` (`:75`), `_sbic_category` (`:95`), `load_sbic` (`:106`), `group_split` (`:138`), `build_all` (`:156`), `load_processed` (`:167`) |
| `biasdetect/ontology/build.py` | **Non** | Aucun test dans `tests/test_modules.py` | `ttl_path` (`:37`), `_uri` (`:57`), `_hb_descriptors` (`:61`), `stereo_attributes` (`:73`), `build` (`:93`) |
| `biasdetect/features/symbolic.py` | **Non** | Aucun test dans `tests/test_modules.py` | `Rule` (`:21`), `SymbolicResult` (`:29`), `Ontology` (`:34`), `SymbolicFeaturizer` (`:65`, `_ngrams`, `analyze`, `transform`) |
| `biasdetect/features/neural.py` | **Non** | Aucun test dans `tests/test_modules.py` | `_get_model` (`:25`), `embed` (`:33`) |
| `biasdetect/models/systems.py` | **Non** | Aucun test dans `tests/test_modules.py` | `System` (`:24`), `Majority` (`:36`), `LegacyRules` (`:48`), `LegacyGenderSim` (`:62`), `ZeroShotNLI` (`:78`), `FineTuned` (`:109`), `LLMJudge` (`:138`), `SymbolicLR` (`:178`), `NeuralProbe` (`:192`), `HybridFusion` (`:208`), `registry` (`:270`) |
| `biasdetect/eval/metrics.py` | **Non** | Aucun test dans `tests/test_modules.py` | `best_threshold` (`:10`), `ece` (`:16`), `metrics` (`:24`), `pairwise_accuracy` (`:38`), `bootstrap_ci` (`:46`), `mcnemar` (`:58`) |
| `biasdetect/llm/client.py` | **Non** | Aucun test dans `tests/test_modules.py` | `_load_dotenv` (`:29`), `load_config` (`:39`), `available_models` (`:44`), `LLMClient` (`:52`, `chat`, `chat_json`) |
| `biasdetect/annotation/export.py` | **Non** | Aucun test dans `tests/test_modules.py` | `main` (`:17`) |
| `biasdetect/annotation/agreement.py` | **Non** | Aucun test dans `tests/test_modules.py` | `load` (`:22`), `main` (`:31`) |

### 4.2 Accès réseau et chargement de modèles dans les tests (`tests/test_modules.py`)
- **Accès réseau dans les tests existants** : **Non**. Aucun test de `tests/test_modules.py:1-57` n'effectue d'appel réseau ni de téléchargement. Si le modèle local `en_core_web_sm` est absent, la fixture saute le test via `pytest.skip("en_core_web_sm not installed")` (`tests/test_modules.py:31`).
- **Chargement de modèles dans les tests existants** : **Oui**. La fixture `sym` charge le modèle spaCy `en_core_web_sm` en mémoire (et ce deux fois de suite) :
  - Preuve 1 : `tests/test_modules.py:29` : `spacy.load("en_core_web_sm")`
  - Preuve 2 : `tests/test_modules.py:33` : `return SymbolicAnalyzer()` qui exécute `self.nlp = spacy.load(model_lang)` à `modules/symbolic.py:11`.
- **Risques d'accès réseau / chargement de modèles dans `biasdetect/` si appelé sans mock** :
  - `biasdetect/data/loaders.py:35` (`urllib.request.urlretrieve`) et `:51` (`load_dataset("McGill-NLP/stereoset", ...)`) téléchargent depuis GitHub/Washington/HuggingFace si les fichiers ne sont pas en cache.
  - `biasdetect/features/neural.py:29` (`SentenceTransformer(ENCODERS[key])`) charge un modèle HuggingFace / SentenceTransformers.
  - `biasdetect/models/systems.py:90-91` (`AutoTokenizer.from_pretrained`, `AutoModelForSequenceClassification.from_pretrained`) et `:157, 161` (`LLMClient` -> API OpenAI/Groq/Gemini à `biasdetect/llm/client.py:68, 80`) chargent un modèle ou appellent une API externe.
