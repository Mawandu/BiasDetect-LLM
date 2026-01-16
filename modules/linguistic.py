from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine
import streamlit as st

class LinguisticAnalyzer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Cache the model loading
        self.model = self._load_model(model_name)
        
        # Define bias subspaces (simplified for demo)
        self.gender_subspace = {
            'male': ['he', 'man', 'boy', 'father', 'son', 'his', 'male'],
            'female': ['she', 'woman', 'girl', 'mother', 'daughter', 'her', 'female']
        }
        
    @st.cache_resource
    def _load_model(_cls, model_name):
        return SentenceTransformer(model_name)

    def analyze_text(self, text):
        """
        Analyzes the text for linguistic bias.
        Returns a dictionary with scores and details.
        """
        if not text.strip():
            return {"score": 0, "details": "Empty text"}
            
        # 1. Embedding Analysis
        # Determine if the text aligns more with male or female subspace
        text_embedding = self.model.encode(text)
        
        male_sim = self._avg_similarity(text_embedding, self.gender_subspace['male'])
        female_sim = self._avg_similarity(text_embedding, self.gender_subspace['female'])
        
        # Calculate bias score (-1 to 1, where 0 is neutral)
        # Positive = Male biased, Negative = Female biased (in this simplistic model)
        # For the final score we might want 0 to 100 where 0 is neutral.
        
        raw_diff = male_sim - female_sim
        
        # Normalize for display: 
        # A raw difference of 0.1 is actually quite significant in embedding space.
        # Let's scale it.
        bias_magnitude = abs(raw_diff) * 10 
        bias_score = min(100, bias_magnitude * 100)
        
        direction = "Neutre"
        if bias_score > 5: # Threshold
            direction = "Masculin" if raw_diff > 0 else "Féminin"
            
        return {
            "score": bias_score,
            "direction": direction,
            "raw_diff": raw_diff,
            "details": {
                "male_similarity": male_sim,
                "female_similarity": female_sim
            }
        }

    def _avg_similarity(self, target_emb, word_list):
        word_embs = self.model.encode(word_list)
        similarities = [1 - cosine(target_emb, w_emb) for w_emb in word_embs]
        return np.mean(similarities)

# Standalone run for testing
if __name__ == "__main__":
    analyzer = LinguisticAnalyzer()
    print(analyzer.analyze_text("The doctor asked the nurse for the chart."))
