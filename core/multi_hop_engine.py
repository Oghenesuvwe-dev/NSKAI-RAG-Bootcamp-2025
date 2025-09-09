import json
import os
from typing import List, Dict
from groq import Groq
from .data_sources import DataSources
from .knowledge_base import KnowledgeBase
from .answer_formatter import AnswerFormatter

class MultiHopRAG:
    def __init__(self):
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.data_sources = DataSources()
        self.knowledge_base = KnowledgeBase()
        self.formatter = AnswerFormatter()
    
    def decompose_query(self, query: str) -> List[str]:
        # Fallback decomposition if API fails
        if "vs" in query.lower() or "compare" in query.lower():
            parts = query.split(" vs ")
            if len(parts) == 2:
                return [f"What is {parts[0].strip()}?", f"What is {parts[1].strip()}?", "How do they compare?"]
        
        try:
            prompt = f"""Break this query into 2-4 sub-questions: "{query}"
            Return JSON array: ["question1", "question2"]"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3
            )
            content = response.choices[0].message.content.strip()
            start = content.find('[')
            end = content.rfind(']') + 1
            if start != -1 and end != 0:
                return json.loads(content[start:end])
        except Exception as e:
            print(f"API Error: {e}")
        
        return [query]
    
    def route_subquery(self, subquery: str) -> str:
        subquery_lower = subquery.lower()
        if any(word in subquery_lower for word in ["stock", "price", "market"]):
            return "finance"
        elif any(word in subquery_lower for word in ["premier", "nba", "sport", "team"]):
            return "sports"
        elif any(word in subquery_lower for word in ["business", "company", "ai"]):
            return "business"
        return "general"
    
    def collect_evidence(self, sub_questions: List[str]) -> List[Dict]:
        evidence = []
        for sq in sub_questions:
            source = self.route_subquery(sq)
            try:
                data = self.data_sources.get_data(sq, source)
                evidence.append({"question": sq, "source": source, "data": data})
            except Exception as e:
                evidence.append({"question": sq, "source": source, "data": f"Error: {str(e)}"})
        return evidence
    
    def synthesize_answer(self, query: str, evidence: List[Dict], context: List[str] = None) -> str:
        # Create fallback answer from evidence
        fallback_answer = f"Based on available data:\n"
        for e in evidence:
            if isinstance(e['data'], dict):
                fallback_answer += f"• {e['source'].title()}: {str(e['data'])}\n"
        
        try:
            evidence_text = "\n".join([f"Q: {e['question']}\nData: {e['data']}" for e in evidence])
            context_text = "\n".join(context) if context else ""
            
            prompt = f"""Answer: {query}
            Evidence: {evidence_text}
            Context: {context_text}
            Provide clear, factual answer with sources."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return fallback_answer + f"\nNote: AI synthesis unavailable, showing raw data."
    
    def process_query(self, query: str) -> Dict:
        sub_questions = self.decompose_query(query)
        evidence = self.collect_evidence(sub_questions)
        context = self.knowledge_base.query_knowledge(query)
        raw_answer = self.synthesize_answer(query, evidence, context)
        return self.formatter.format_response(raw_answer, evidence, sub_questions)