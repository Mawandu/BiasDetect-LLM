# BiasDetect-LLM

**Système Hybride de Détection de Biais dans les LLM**

Ce démonstrateur analyse les textes générés par des LLM pour détecter des biais de genre, culturels et de représentation en utilisant une approche hybride innovante combinant l'analyse neuronale et symbolique.

## Architecture et Fonctionnement

Le système repose sur une double analyse pour garantir à la fois la détection de biais subtils (implicites) et la vérification de règles normatives (explicites).

```
┌─────────────────────────────────────────────────┐
│         INPUT : Texte généré par LLM            │
└─────────────────┬───────────────────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
┌───────▼────────┐  ┌──────▼────────┐
│   ANALYSE      │  │  ONTOLOGIES   │
│  LINGUISTIQUE  │  │  DE DOMAINE   │
│                │  │               │
│ • Embeddings   │  │ • Genre       │
│ • WEAT/SEAT    │  │ • Culture     │
│ • Contrastes   │  │ • Représent.  │
└───────┬────────┘  └──────┬────────┘
        │                   │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  FUSION HYBRIDE   │
        │   (60% + 40%)     │
        └─────────┬─────────┘
                  │
        ┌─────────▼─────────┐
        │  SCORING +        │
        │  EXPLICABILITÉ    │
        └─────────┬─────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│  OUTPUT : Rapport détaillé + Visualisations     │
│  • Scores par dimension de biais                │
│  • Justifications linguistiques                 │
│  • Contexte sémantique (ontologie)              │
│  • Suggestions de mitigation                    │
└─────────────────────────────────────────────────┘
```

### Méthodologie

Nous procédons en trois étapes clés :

1.  **Analyse Linguistique (Approche Neuronale)** : Nous utilisons des modèles de langage (Sentence-Transformers) pour projeter le texte dans un espace vectoriel. Nous mesurons ensuite la distance cosinus entre le texte et des concepts biaisés (ex: masculin vs féminin) pour identifier des associations implicites.
2.  **Analyse Symbolique (Approche Ontologique)** : Nous extrayons les entités nommées (personnes, lieux, professions) avec Spacy et les confrontons à une base de connaissances (ontologie). Cela permet de détecter des stéréotypes explicites (ex: "Infirmier" est une profession neutre, mais souvent stéréotypée féminine) ou des clichés culturels.
3.  **Fusion et Explicabilité** : Les deux scores sont fusionnés via une pondération ajustable ($\alpha$). Le système génère ensuite un rapport détaillé expliquant *pourquoi* un biais a été détecté, en surlignant les termes problématiques et en indiquant le type de stéréotype (Genre, Culture, Représentation).

## Installation des dépendances

Les versions d'exécution sont épinglées dans `requirements.txt` ; les outils de test sont dans `requirements-dev.txt`. `torch` n'est pas listé explicitement : il est installé comme dépendance de `sentence-transformers`.

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
python -m spacy download en_core_web_sm
```

La dernière commande télécharge le modèle spaCy `en_core_web_sm` (une seule fois, avec accès réseau) ; elle n'est pas exécutée automatiquement.

## Utilisation

```bash
streamlit run main.py
```
