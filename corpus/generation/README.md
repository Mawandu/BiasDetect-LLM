# Génération du corpus BiasDetect-Corpus

Script, prompts et sorties brutes des 432 textes (24 prompts × 6 modèles × 3 générations).

- `prompts.py` : les 24 prompts (GENDER, ORIGIN, RELIGION, OTHER).
- `config.py` : modèles, température (0,8), nombre de générations par prompt (3),
  limite de tokens par défaut (120).
- `generate.py` : lance la génération ; `utils.py` : appels API avec reprises ; la limite passe à 512 tokens pour les modèles de raisonnement (définie dans `utils.py`).
- `generated_raw.csv` : sorties brutes. SHA-256 :
  5cbc11b0f626391b16f81eed228e724a014dd3e8fb07961f07056886a9d69954
- Clés d'API : copier `api_keys.env.example` en `api_keys.env` et renseigner
  OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY.
  Ce fichier est ignoré par git et ne doit jamais être publié.

Note : les sorties dépendent des versions des modèles au moment de l'appel
(24 octobre 2026) ; relancer le script ne redonnera pas exactement les mêmes textes.
