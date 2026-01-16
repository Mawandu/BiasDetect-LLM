class FusionEngine:
    def __init__(self):
        pass
        
    def fuse_scores(self, neural_res, symbolic_res, alpha=0.6):
        """
        Combines scores from both branches.
        alpha: Weight for Neural branch (0.0 to 1.0)
        """
        neural_score = neural_res.get('score', 0)
        symbolic_score = symbolic_res.get('score', 0)
        
        # Weighted avg
        final_score = (alpha * neural_score) + ((1 - alpha) * symbolic_score)
        
        return {
            "final_score": round(final_score, 2),
            "sources": {
                "neural": neural_score,
                "symbolic": symbolic_score,
                "alpha": alpha
            },
            "explanation": self.generate_explanation(neural_res, symbolic_res, final_score)
        }
        
    def generate_explanation(self, neural_res, symbolic_res, final_score):
        """
        Generates a human-readable explanation of the bias.
        """
        parts = []
        
        # Neural part
        if neural_res.get('score', 0) > 10:
            direction = neural_res.get('direction', 'detected')
            parts.append(f"L'analyse linguistique a détecté un biais potentiellement **{direction}** (Score: {neural_res['score']:.1f}).")
        
        # Symbolic part
        issues = symbolic_res.get('issues', [])
        if issues:
            terms = ", ".join([f"'{i['term']}'" for i in issues])
            parts.append(f"L'analyse symbolique a identifié des termes stéréotypés : {terms}.")
        
        if final_score < 10:
            return "Aucun biais significatif détecté par le système hybride."
            
        summary = " ".join(parts)
        return f"**Biais Détecté** (Score Global: {final_score:.1f}/100) : {summary}"

if __name__ == "__main__":
    eng = FusionEngine()
    print(eng.fuse_scores({"score": 60, "direction": "Masculin"}, {"score": 20, "issues": []}))
