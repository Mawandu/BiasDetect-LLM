# Plan d'analyse — BiasDetect-Corpus

- **Date** : 2026-10-28
- **Commit du corpus** : `434b433` (`434b43393a38b74f72fca5a19eaa7af21ee6ea2d`)
- **Contenu de `corpus/SHA256SUMS`** :

```text
81bf3bd1b8854677bd4f7abbc7582001e8bb81ea8e6b5c98c9b8969f819b4ea4  README.md
3057d4644c831b080efc2ade25dd3c5bf3d7d57c0879b2129d25c9d9cfbbfce7  agreement_summary_round1.csv
b24ad28bdad27e86509941fa5110e50d62a09558c8c07f21d080ce22e0706c8c  biasdetect_corpus.csv
```

Ce plan est fixé avant toute évaluation ; aucun résultat d'évaluation n'y figure.

## 1. Jeu de test

Jeu de test : `corpus/biasdetect_corpus.csv`. Le gold principal est `gold_raw` ; la sensibilité est `gold_reviewed`.

## 2. Sous-ensembles

Sous-ensembles analysés : tout le corpus ; Core (`is_core`) pour l'analyse principale ; OTHER pour l'analyse exploratoire ; hors refus (`refusal` = False).

## 3. Systèmes

Systèmes comparés : Majority, LegacyRules, LegacyGenderSim, P_sym, P_neu_*, P_fusion_* (registre rapide). LLMJudge et ZeroShotNLI sont exclus (coût).

## 4. Entraînement et seuil

Entraînement sur StereoSet train ; seuil choisi sur StereoSet dev (jamais sur le corpus) ; 5 seeds : 13, 21, 42, 7, 99.

## 5. Modes de score

Deux modes de score fixés d'avance : `full` (texte entier) et `sentmean` (moyenne des scores par phrase, découpage par regex). Aucun choix entre les deux ne sera fait d'après les résultats sur le corpus : les deux sont rapportés.

## 6. Métriques

Métriques : AUROC (principale, indépendante du seuil), macro-F1 au seuil dev, ECE (information).

## 7. Incertitude

Incertitude : bootstrap en grappes sur `prompt_id` (24 prompts, 2000 rééchantillonnages), avec la moyenne des métriques sur les seeds à l'intérieur de chaque rééchantillon ; différences appariées (P_fusion - P_sym, P_fusion - P_neu) avec le même rééchantillon. McNemar en complément (informatif : ignore les grappes).

## 8. Hypothèse annoncée

Hypothèse annoncée : l'hybride n'est PAS supposé meilleur ; tout résultat sera rapporté.

## 9. Critère principal et multiplicité

- Analyse principale : sous-ensemble Core (GENDER, ORIGIN, RELIGION) hors refus (`refusal` = False), gold `gold_raw`, métrique AUROC. Les refus sont des textes trivialement non stéréotypés : les inclure gonflerait l'AUROC de tout système qui note bas les textes courts. Les résultats avec refus sont rapportés en analyse secondaire.
- Comparaisons principales : P_fusion_minilm contre P_neu_minilm, et P_fusion_minilm contre P_sym, dans les deux modes de score (`full`, `sentmean`), soit 4 comparaisons. Correction de Holm au niveau 0.05 sur ces 4 comparaisons. Toutes les autres analyses sont exploratoires et présentées comme telles.

## 10. Robustesse du gold

Les analyses principales sont répétées avec : (a) `gold_reviewed` ; (b) un gold restreint aux items sur lesquels A1 et A2 s'accordent (exclut A3) ; (c) sans les textes de moins de 15 mots.

## 11. Définitions techniques

- Découpage en phrases : regex `(?<=[.!?])\s+`.
- Graine du bootstrap fixée à 13. Un rééchantillon ne contenant qu'une seule classe est écarté et compté dans le rapport.
- Le seuil de macro-F1 est celui choisi sur StereoSet dev pour chaque seed.
- Tout écart à ce plan sera signalé dans la section « Écarts au plan » de l'article.

## 12. Précisions de mise en œuvre (rédigées le 2026-10-29, avant tout résultat)
- Bootstrap en grappes : les prompts sont rééchantillonnés à l'intérieur du sous-ensemble évalué
  (tout = 24 prompts ; Core = 17), grappes complètes, avec le même tirage pour tous les systèmes,
  modes et golds d'un sous-ensemble. Un rééchantillon à une seule classe est rejeté et compté
  (colonne n_boot_rejected). Les métriques sont moyennées sur les 5 seeds à l'intérieur de
  chaque rééchantillon.
- p bootstrap bilatérale = min(1, 2 (min(k+, k-) + 1) / (B + 1)) ; correction de Holm sur les
  4 comparaisons principales (2 paires × 2 modes). McNemar : informatif, par seed.
- ECE : rapporté seulement si tous les scores du sous-ensemble sont dans [0, 1], sinon « n/a ».
- Contrôle de cohérence : exécution avec étiquettes permutées (--permute-gold), AUROC attendu ~ 0,5.
Résultats produits au commit 0a8a492 (renommé ecd5d93 par amend de message, arbre identique)

## 13. Analyses exploratoires (post hoc, rédigées le 2026-10-29 APRÈS avoir vu les résultats principaux)
Les résultats principaux (AUROC Core, gold_raw) sont inférieurs à 0,5 pour la plupart des systèmes.
Les analyses suivantes cherchent à comprendre ce résultat ; elles ne remplacent pas l'analyse principale
et sont rapportées comme exploratoires.
- E1 : décomposition entre prompts / à l'intérieur d'un prompt. AUROC moyen intra-prompt (prompts
  Core hors refus ayant les deux classes) et corrélation de Spearman entre le score moyen d'un prompt
  et son taux de textes étiquetés 1. Scores moyennés sur les 5 seeds.
- Aucune étiquette n'est modifiée, aucun seuil n'est réglé sur le corpus.
