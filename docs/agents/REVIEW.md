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

---

# Revue de Code — Tâche 4 (`9204994..HEAD`)

**Statut : OK**

## 1. Vérification du diff et du critère
- **`biasdetect/annotation/agreement.py:8`** : la docstring documente désormais `data/bold/gold.csv` (alignée sur le chemin réel).
- **`biasdetect/annotation/agreement.py:64-66`** : `out_path = ROOT / "data" / "bold" / "gold.csv"` est précédé de `out_path.parent.mkdir(parents=True, exist_ok=True)` avant `out.to_csv(out_path)`.
- **`tests/test_biasdetect.py:71-99` (`test_agreement_resolve_creates_parent_and_writes_gold_in_tmp_path`)** :
  - Redirige `agreement.ROOT` et `agreement.ANN` vers `tmp_path` (`tests/test_biasdetect.py:87-88`) : aucune écriture dans `/data/`, aucun accès réseau, aucun chargement de modèle.
  - Vérifie que `tmp_path / "data" / "bold" / "gold.csv"` et son dossier parent sont créés et contiennent les 2 lignes attendues (`[0, 1]`).
  - Sans `out_path.parent.mkdir(parents=True, exist_ok=True)` (`biasdetect/annotation/agreement.py:65`), `out.to_csv(out_path)` échoue avec `OSError` car `tmp_path / "data" / "bold"` n'existe pas avant l'appel.
- **`docs/agents/TASKS.md:26`** : la Tâche 4 est marquée `[x]`.

---

# Revue de Code — Tâche 6 (`9204994..HEAD`)

**Statut : OK**

## 1. Vérification du diff et des critères
- **`tests/test_biasdetect.py:103-214`** :
  - Couvre toutes les fonctions de `biasdetect/eval/metrics.py` : `best_threshold` (`:103-108`), `ece` (`:111-113`), `metrics` (`:122-140`, cas parfait, `NaN`, classe unique, scores hors `[0, 1]`), `pairwise_accuracy` (`:143-149`), `bootstrap_ci` (`:152-158`) et `mcnemar` (`:161-166`).
  - Couvre `_sbic_category` (`:171-181`, 7 cas paramétrés) et `group_split` (`:184-214`, déterminisme, stratification par catégorie et disjonction stricte des `group_id` entre `train`, `dev` et `test`).
  - Aucun test n'écrit dans `data/`, ne fait d'accès réseau ni ne charge de modèle. Tous les appels de `tests/test_biasdetect.py` s'exécutent en moins de 2 secondes (le plus lent est `test_best_threshold_separates_perfectly` en `0.86s`).
- **`docs/agents/TASKS.md:40`** : la Tâche 6 est marquée `[x]`.

## 2. Recalcul manuel des valeurs attendues
- **`pairwise_accuracy` (`tests/test_biasdetect.py:143-149`, `biasdetect/eval/metrics.py:38-44`)** :
  - 3 paires `group_id` $\in \{\text{"a"}, \text{"b"}, \text{"c"}\}$ avec `label` $\in \{0, 1\}$ et scores `s = [0.2, 0.8, 0.5, 0.5, 0.9, 0.1]` :
    - Paire `"a"` : $s_1 = 0.8 > s_0 = 0.2 \implies 1.0$
    - Paire `"b"` : $s_1 = 0.5 = s_0 = 0.5$ (ex-aequo compté $0.5$) $\implies 0.5$
    - Paire `"c"` : $s_1 = 0.1 < s_0 = 0.9 \implies 0.0$
  - Moyenne sur les 3 paires : $\frac{1.0 + 0.5 + 0.0}{3} = \frac{1.5}{3} = 0.5$, conforme à `pytest.approx(0.5)` (`tests/test_biasdetect.py:149`).
- **`mcnemar` (`tests/test_biasdetect.py:161-166`, `biasdetect/eval/metrics.py:58-62`)** :
  - `y = [0, 0, 0, 0]`, `same = [0, 0, 0, 0]`, `wrong = [1, 1, 1, 1]`.
  - Cas identique (`mcnemar(y, same, same)`) : $b = 0, c = 0 \implies b + c = 0 \implies 1.0$ (`tests/test_biasdetect.py:165`).
  - Cas discordant (`mcnemar(y, same, wrong)`) : `a_ok = [T, T, T, T]`, `b_ok = [F, F, F, F]`, d'où $b = 4$ et $c = 0$ ($b + c = 4$). Sous $\mathcal{B}(4, 0.5)$, la p-valeur bilatérale exacte vaut $P(X = 0) + P(X = 4) = (0.5)^4 + (0.5)^4 = 0.0625 + 0.0625 = 0.125$, conforme à `pytest.approx(0.125)` (`tests/test_biasdetect.py:166`).

---

# Revue de Code — Tâche 11 (`9204994..HEAD`)

**Statut : OK**

## 1. Vérification du diff et recalcul manuel de `ece`
- **`biasdetect/eval/metrics.py`** : non modifié.
- **`tests/test_biasdetect.py:116-120` (`test_ece_weights_bins_by_frequency`)** :
  - Entrées : `y = np.array([0, 0, 0, 1])`, `p = np.array([0.15, 0.15, 0.15, 0.85])`, `bins = 10` (tranches de largeur $0.1$).
  - Recalcul manuel :
    - Tranche $b = 1$ ($[0.1, 0.2)$) : contient 3 exemples avec $y = [0, 0, 0]$ et $p = [0.15, 0.15, 0.15]$. Fréquence $= \frac{3}{4} = 0.75$, écart $|\bar{y}_1 - \bar{p}_1| = |0.0 - 0.15| = 0.15$, contribution $= 0.15 \times \frac{3}{4} = 0.1125$.
    - Tranche $b = 8$ ($[0.8, 0.9)$) : contient 1 exemple avec $y = [1]$ et $p = [0.85]$. Fréquence $= \frac{1}{4} = 0.25$, écart $|\bar{y}_8 - \bar{p}_8| = |1.0 - 0.85| = 0.15$, contribution $= 0.15 \times \frac{1}{4} = 0.0375$.
    - Somme pondérée : $\text{ECE} = 0.1125 + 0.0375 = 0.15$, conforme à `pytest.approx(0.15)` (`tests/test_biasdetect.py:119`).
  - Si `* (idx == b).mean()` était remplacé par `* 1` dans `biasdetect/eval/metrics.py:20`, la somme sur les deux tranches non vides vaudrait $0.15 + 0.15 = 0.30 \neq 0.15$, faisant échouer le test.
- **`docs/agents/TASKS.md:73`** : la Tâche 11 est marquée `[x]`.

## 2. Résultat des tests (`python -m pytest -q --durations=5`)
```text
.................................                                        [100%]
============================= slowest 5 durations ==============================
5.55s setup    tests/test_modules.py::test_symbolic_single_word
0.86s call     tests/test_biasdetect.py::test_best_threshold_separates_perfectly
0.37s call     tests/test_biasdetect.py::test_bootstrap_ci_deterministic_on_perfect_scores
0.20s call     tests/test_biasdetect.py::test_symbolic_lr_passes_seed_to_featurizer
0.19s call     tests/test_biasdetect.py::test_agreement_resolve_creates_parent_and_writes_gold_in_tmp_path
33 passed in 45.54s
```

Aucun problème détecté.

---

# Revue de Code — Tâche 7a (`b57ad0c..HEAD`)

**Statut : OK**

## 1. Vérification du diff et des critères
- **`tests/test_ontology.py:1-184` (nouveau fichier)** :
  - La fixture `mini_ttl` (`tests/test_ontology.py:60-69`) sérialise une mini-ontologie RDF/Turtle dans `tmp_path / "mini.ttl"`, redirige `symbolic.ttl_path` vers ce fichier via `monkeypatch` (`:67`) et remplace `symbolic.build` par `fail_build` (`:68`) qui lève `AssertionError` s'il est appelé. Aucun test ne lit ni n'écrit `data/ontology/bias_s13.ttl` ni aucun fichier de `data/`, et aucun accès réseau ni chargement de modèle n'a lieu.
  - Couvre `biasdetect/ontology/build.py` : `_uri` (`:79-82`), `_hb_descriptors` (`:85-98`), `stereo_attributes` avec `load_processed` mocké (`:101-128`).
  - Couvre `biasdetect/features/symbolic.py` : `Ontology` (`:133-146`), `SymbolicFeaturizer.analyze` (`:149-173`, règles `group_mention`, `dispreferred_term` multi-mots, `occ_congruent`, `stereo_attribute`) et `SymbolicFeaturizer.transform` (`:176-184`). Casser l'une des 3 branches de règles dans `SymbolicFeaturizer.analyze` (`biasdetect/features/symbolic.py:86-129`) fait échouer `test_analyze_group_occupation_and_attribute_rules` ou `test_analyze_dispreferred_multiword_term`.
- **`docs/agents/TASKS.md:78`** : la Tâche 7a est marquée `[x]`.

## 2. Recalcul manuel des valeurs attendues
- **`stereo_attributes` (`tests/test_ontology.py:101-128`, `biasdetect/ontology/build.py:73-90`)** :
  - Sur le split `train`, il y a $n_1 = 3$ phrases positives (`"happy family"` $\times 3$) et $n_0 = 2$ phrases négatives (`"sad person"` $\times 2$) ; la 6ᵉ ligne (`"zzz test"`, `split == "test"`) est exclue.
  - Pour `"happy"` : $c_1 = 3, c_0 = 0, \alpha = 1.0 \implies \text{log\_odds} = \ln\left(\frac{3+1}{3-3+1}\right) - \ln\left(\frac{0+1}{2-0+1}\right) = \ln(4) - \ln(1/3) = \ln(12)$, conforme à `pytest.approx(math.log(12))` (`tests/test_ontology.py:120`).
  - Pour `"sad"` (`min_count=2`) : $c_1 = 0, c_0 = 2, \alpha = 1.0 \implies \text{log\_odds} = \ln\left(\frac{0+1}{3-0+1}\right) - \ln\left(\frac{2+1}{2-2+1}\right) = \ln(1/4) - \ln(3) = -\ln(12)$, conforme à `pytest.approx(-math.log(12))` (`tests/test_ontology.py:127`).
- **`SymbolicFeaturizer.analyze` (`tests/test_ontology.py:149-163`, `biasdetect/features/symbolic.py:80-129`)** :
  - Phrase : `"The woman is a nurse and she is emotional."`
  - Métier `"nurse"` avec `pctFemale = 90.0` (`tests/test_ontology.py:52`) : $\text{skew} = \frac{90.0 - 50}{50} = 0.8 \implies \text{occ\_skew} = |0.8| = 0.8$ (`tests/test_ontology.py:154`). Comme $|\text{skew}| = 0.8 \ge 0.4$, `stereo_pole = "female"` et la phrase contient uniquement des références féminines (`"woman"`, `"she"`), d'où `occ_congruent = 1` et `occ_incongruent = 0` (`tests/test_ontology.py:155-156`).
  - Attribut `"emotional"` (`logOdds = 1.5` pour `"gender"`, `tests/test_ontology.py:54`) : seul attribut présent dans la phrase, d'où `attr_max_gender = 1.5`, `attr_min_gender = 1.5`, `attr_sum = 1.5` (`tests/test_ontology.py:157-159`).

## 3. Résultat des tests (`python -m pytest -q --durations=5`)
```text
........................................                                 [100%]
============================= slowest 5 durations ==============================
3.93s setup    tests/test_modules.py::test_symbolic_single_word
1.44s call     tests/test_biasdetect.py::test_best_threshold_separates_perfectly
1.04s call     tests/test_biasdetect.py::test_agreement_resolve_creates_parent_and_writes_gold_in_tmp_path
0.55s call     tests/test_biasdetect.py::test_bootstrap_ci_deterministic_on_perfect_scores
0.15s call     tests/test_biasdetect.py::test_symbolic_lr_passes_seed_to_featurizer
40 passed in 55.33s
```

Aucun problème détecté.
