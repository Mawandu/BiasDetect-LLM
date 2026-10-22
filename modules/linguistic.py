from sentence_transformers import SentenceTransformer
import numpy as np


class LinguisticAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Model caching is handled by the caller (see main.load_analyzers)
        self.model = SentenceTransformer(model_name)

        # Define bias subspaces (simplified for demo)
        self.gender_subspace = {
            'male': ['he', 'man', 'boy', 'father', 'son', 'his', 'male'],
            'female': ['she', 'woman', 'girl', 'mother', 'daughter', 'her', 'female']
        }

        # Pre-compute (normalized) embeddings of the reference words once
        self._subspace_embs = {
            key: self._normalize(self.model.encode(words))
            for key, words in self.gender_subspace.items()
        }

    @staticmethod
    def _normalize(vectors):
        vectors = np.atleast_2d(vectors)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def analyze_text(self, text):
        """
        Analyzes the text for linguistic bias.
        Returns a dictionary with scores and details.
        """
        if not text or not text.strip():
            return {"score": 0.0, "direction": "Neutre", "raw_diff": 0.0, "details": "Empty text"}

        # 1. Embedding Analysis
        # Determine if the text aligns more with male or female subspace
        text_emb = self._normalize(self.model.encode(text))[0]

        male_sim = self._avg_similarity(text_emb, 'male')
        female_sim = self._avg_similarity(text_emb, 'female')

        # Positive = Male biased, Negative = Female biased (in this simplistic model)
        raw_diff = male_sim - female_sim

        # Normalize to 0-100 with a saturating curve: small differences in
        # embedding space are meaningful, but the score should not explode.
        # |diff| = 0.05 -> ~46, 0.1 -> ~76, 0.2 -> ~96
        bias_score = float(100 * np.tanh(abs(raw_diff) * 10))

        direction = "Neutre"
        if bias_score > 5:  # Threshold
            direction = "Masculin" if raw_diff > 0 else "Féminin"

        return {
            "score": round(bias_score, 2),
            "direction": direction,
            "raw_diff": round(raw_diff, 4),
            "details": {
                "male_similarity": round(male_sim, 4),
                "female_similarity": round(female_sim, 4)
            }
        }

    def _avg_similarity(self, text_emb, key):
        # Cosine similarity = dot product of normalized vectors
        return float(np.mean(self._subspace_embs[key] @ text_emb))


# Standalone run for testing
if __name__ == "__main__":
    analyzer = LinguisticAnalyzer()
    print(analyzer.analyze_text("The doctor asked the nurse for the chart."))
