import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
import os

class KnowledgeBase:
    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path="./chroma_db")
            self.collection = self.client.get_or_create_collection("knowledge")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.load_documents()
        except:
            self.client = None
            self.collection = None
    
    def load_documents(self):
        if not self.collection:
            return
            
        documents = {
            "sports_rules": "Premier League has 20 teams, top 4 qualify for Champions League. NBA has 30 teams in Eastern and Western conferences.",
            "finance_basics": "Stock prices reflect company valuation. Market cap = shares × price. P/E ratio shows valuation relative to earnings.",
            "business_analysis": "AI market growing 36.8% CAGR. Microsoft invested $13B in OpenAI. Tech companies focus on AI integration."
        }
        
        try:
            for doc_id, content in documents.items():
                embedding = self.embedder.encode(content)
                self.collection.add(
                    documents=[content],
                    embeddings=[embedding.tolist()],
                    ids=[doc_id]
                )
        except:
            pass
    
    def query_knowledge(self, question: str, n_results: int = 2) -> List[str]:
        if not self.collection:
            return ["Basic knowledge available"]
            
        try:
            query_embedding = self.embedder.encode(question)
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except:
            return ["Knowledge base unavailable"]