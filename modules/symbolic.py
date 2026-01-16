import spacy
from typing import Dict, List, Any

class SymbolicAnalyzer:
    def __init__(self, model_lang='en_core_web_sm'):
        try:
            self.nlp = spacy.load(model_lang)
        except OSError:
            # Fallback or instruction to download
            # In a real app we'd download it. For now assume it's present or handle error.
            print(f"Warning: Model {model_lang} not found. Please run 'python -m spacy download {model_lang}'")
            self.nlp = None

        # Mock Ontology: Rules about professions and expected neutrality vs stereotypes
        self.gender_rules = {
            "doctor": {"type": "profession", "expected": "neutral", "stereotype": "male"},
            "nurse": {"type": "profession", "expected": "neutral", "stereotype": "female"},
            "engineer": {"type": "profession", "expected": "neutral", "stereotype": "male"},
            "teacher": {"type": "profession", "expected": "neutral", "stereotype": "female"},
            "pilot": {"type": "profession", "expected": "neutral", "stereotype": "male"},
            "flight attendant": {"type": "profession", "expected": "neutral", "stereotype": "female"},
            "ceo": {"type": "profession", "expected": "neutral", "stereotype": "male"},
            "secretary": {"type": "profession", "expected": "neutral", "stereotype": "female"},
        }
        
        # Cultural & Geographic Stereotypes
        self.cultural_rules = {
            "africa": {"stereotype": "poverty/instability", "sensitive": True},
            "asia": {"stereotype": "math/tech efficient", "sensitive": True},
            "latino": {"stereotype": "fiery/loud", "sensitive": True},
            "paris": {"stereotype": "romantic", "sensitive": False}, # Less harmful but still a cliche
        }
        
        # Representation & Sensitive Attributes
        self.representation_rules = {
            "ghetto": {"category": "socioeconomic", "severity": "high"},
            "illegal alien": {"category": "immigration", "severity": "critial", "suggestion": "undocumented immigrant"},
            "mankind": {"category": "gender-exclusive", "severity": "low", "suggestion": "humankind"},
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities and checks against ontological rules.
        """
        if not self.nlp:
            return {"score": 0, "details": "NLP Model not loaded", "issues": []}
            
        doc = self.nlp(text)
        
        detected_issues = []
        score_impact = 0
        
        # Analyze nouns and entities
        normalized_text = text.lower()
        
        for token in doc:
            lemma = token.lemma_.lower()
            text_lower = token.text.lower()
            
            # 1. Gender/Profession Logic
            if lemma in self.gender_rules:
                rule = self.gender_rules[lemma]
                detected_issues.append({
                    "term": token.text,
                    "concept": lemma,
                    "type": "Gender Stereotype",
                    "stereotype_risk": rule['stereotype'],
                    "explanation": f"The term '{token.text}' has a known {rule['stereotype']} stereotype in the ontology."
                })
                score_impact += 10
            
            # 2. Cultural/Geographic Logic
            if lemma in self.cultural_rules or text_lower in self.cultural_rules:
                key = lemma if lemma in self.cultural_rules else text_lower
                rule = self.cultural_rules[key]
                detected_issues.append({
                    "term": token.text,
                    "concept": key,
                    "type": "Cultural Stereotype",
                    "stereotype_risk": rule['stereotype'],
                    "explanation": f"Potential cultural overgeneralization or stereotype: '{rule['stereotype']}'."
                })
                score_impact += 15 # Higher weight for cultural sensitivity
                
            # 3. Representation/Sensitivity Logic
            if text_lower in self.representation_rules or lemma in self.representation_rules:
                key = text_lower if text_lower in self.representation_rules else lemma
                rule = self.representation_rules[key]
                msg = f"Sensitive term detected (severity: {rule['severity']})."
                if 'suggestion' in rule:
                    msg += f" Consider using '{rule['suggestion']}'."
                
                detected_issues.append({
                    "term": token.text,
                    "concept": key,
                    "type": "Representation",
                    "stereotype_risk": rule['category'],
                    "explanation": msg
                })
                score_impact += 20 if rule['severity'] in ['high', 'critial'] else 5

        # Normalize score 0-100
        final_score = min(100, score_impact)

        return {
            "score": final_score,
            "issues": detected_issues,
            "entity_count": len(doc.ents)
        }

if __name__ == "__main__":
    analyzer = SymbolicAnalyzer()
    if analyzer.nlp:
        print(analyzer.analyze_text("The doctor said he was busy."))
