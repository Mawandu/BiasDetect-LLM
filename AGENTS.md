# Règles communes (tous les agents)
- Lis ce fichier et docs/agents/ avant d'agir. Respecte ton rôle.
- Une tâche à la fois, sur ta branche. Jamais sur main ni sur wip/avant-agents. Ne fusionne pas : c'est l'humain qui fusionne.
- N'invente rien : aucune source, chiffre, date ni résultat. Si tu ne sais pas, écris « non vérifié ».
- Ne lis, n'affiche et n'écris jamais de clé d'API. Ne modifie pas .env.
- Ne lance aucun script qui appelle une API externe (Groq, OpenAI, Gemini) ni aucun téléchargement. Les tests ne doivent faire aucun appel réseau (utilise des mocks).
- data/ et results/ sont en lecture seule.
- Environnement : source .venv/bin/activate. Tests : python -m pytest -q

# Rôles
- Auditeur/testeur (Antigravity) : examine, lance les tests, relit les diffs, écrit dans docs/agents/AUDIT.md et REVIEW.md. Ne modifie pas le code source.
- Correcteur (OpenCode) : corrige selon docs/agents/TASKS.md, une tâche à la fois, sur la branche agent/fix.

# Circuit
1. Auditeur : AUDIT.md (ce qui marche, ce qui ne marche pas, anomalies avec fichier:ligne et preuve) et propose TASKS.md.
2. Humain : relit, corrige, valide.
3. Correcteur : fait une tâche, lance les tests, commit, marque [x].
4. Testeur : REVIEW.md (OK, ou problèmes précis). S'il y a un problème, il ajoute une tâche [ ] dans TASKS.md pour le correcteur.
5. On boucle jusqu'à OK. L'humain fusionne.
