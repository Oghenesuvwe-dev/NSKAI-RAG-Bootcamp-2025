from typing import Dict, List

class AnswerFormatter:
    def format_response(self, raw_answer: str, evidence: List[Dict], sub_questions: List[str]) -> Dict:
        sources = list(set([e["source"] for e in evidence]))
        confidence = self.calculate_confidence(evidence)
        
        return {
            "answer": self.clean_answer(raw_answer),
            "sources": sources,
            "confidence": confidence,
            "sub_questions": sub_questions,
            "evidence": evidence
        }
    
    def clean_answer(self, answer: str) -> str:
        return answer.replace("```", "").strip()
    
    def calculate_confidence(self, evidence: List[Dict]) -> float:
        base = 0.7
        if len(evidence) >= 3: base += 0.1
        if any(e["source"] in ["finance", "sports"] for e in evidence): base += 0.1
        return min(base, 0.95)