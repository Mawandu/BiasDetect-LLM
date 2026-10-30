import spacy
from spacy.matcher import PhraseMatcher
from typing import Dict, Any

HIGH_SEVERITY = {"high", "critical"}


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
            "paris": {"stereotype": "romantic", "sensitive": False},  # Less harmful but still a cliche
        }

        # Representation & Sensitive Attributes
        self.representation_rules = {
            "ghetto": {"category": "socioeconomic", "severity": "high"},
            "illegal alien": {"category": "immigration", "severity": "critical", "suggestion": "undocumented immigrant"},
            "mankind": {"category": "gender-exclusive", "severity": "low", "suggestion": "humankind"},
        }

        if self.nlp:
            self._build_matchers()

    def _build_matchers(self):
        """
        Builds phrase matchers so that multi-word concepts
        (e.g. 'flight attendant', 'illegal alien') are detected.
        Two matchers: one on the lowercase form, one on lemmas
        (to catch plurals such as 'nurses' or 'illegal aliens').
        """
        self.matcher_lower = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher_lemma = PhraseMatcher(self.nlp.vocab, attr="LEMMA")
        for label, rules in (("GENDER", self.gender_rules),
                             ("CULTURAL", self.cultural_rules),
                             ("REPRESENTATION", self.representation_rules)):
            for term in rules:
                self.matcher_lower.add(f"{label}|{term}", [self.nlp.make_doc(term)])
                # Lemma patterns need the full pipeline; lemmas are lowercase for these terms
                self.matcher_lemma.add(f"{label}|{term}", [self.nlp(term)])

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Extracts entities and checks against ontological rules.
        """
        if not self.nlp:
            return {"score": 0, "details": "NLP Model not loaded", "issues": []}
        if not text or not text.strip():
            return {"score": 0, "issues": [], "entity_count": 0}

        doc = self.nlp(text)

        # Collect matches from both matchers, de-duplicated by span position
        found = {}
        for matcher in (self.matcher_lower, self.matcher_lemma):
            for match_id, start, end in matcher(doc):
                found[(start, end)] = self.nlp.vocab.strings[match_id]

        # Drop spans contained in a longer match (e.g. 'alien' inside 'illegal alien')
        spans = sorted(found, key=lambda s: (s[0], -(s[1] - s[0])))
        kept, last_end = [], -1
        for start, end in spans:
            if end <= last_end:
                continue
            kept.append((start, end))
            last_end = end

        detected_issues = []
        score_impact = 0

        for start, end in kept:
            label, key = found[(start, end)].split("|", 1)
            term = doc[start:end].text

            # 1. Gender/Profession Logic
            if label == "GENDER":
                rule = self.gender_rules[key]
                detected_issues.append({
                    "term": term,
                    "concept": key,
                    "type": "Gender Stereotype",
                    "stereotype_risk": rule['stereotype'],
                    "explanation": f"The term '{term}' has a known {rule['stereotype']} stereotype in the ontology."
                })
                score_impact += 10

            # 2. Cultural/Geographic Logic
            elif label == "CULTURAL":
                rule = self.cultural_rules[key]
                detected_issues.append({
                    "term": term,
                    "concept": key,
                    "type": "Cultural Stereotype",
                    "stereotype_risk": rule['stereotype'],
                    "explanation": f"Potential cultural overgeneralization or stereotype: '{rule['stereotype']}'."
                })
                score_impact += 15 if rule['sensitive'] else 5  # Higher weight for cultural sensitivity

            # 3. Representation/Sensitivity Logic
            elif label == "REPRESENTATION":
                rule = self.representation_rules[key]
                msg = f"Sensitive term detected (severity: {rule['severity']})."
                if 'suggestion' in rule:
                    msg += f" Consider using '{rule['suggestion']}'."

                detected_issues.append({
                    "term": term,
                    "concept": key,
                    "type": "Representation",
                    "stereotype_risk": rule['category'],
                    "explanation": msg
                })
                score_impact += 20 if rule['severity'] in HIGH_SEVERITY else 5

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
