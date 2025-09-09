import os
from core.multi_hop_engine import MultiHopRAG

# Test the RAG system
def test_rag():
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY environment variable")
        return
    
    rag = MultiHopRAG()
    
    test_queries = [
        "How did Liverpool's performance affect Champions League chances vs Arsenal?",
        "Microsoft AI investment impact vs Google strategy on stock prices",
        "Tech stock and sports team valuation correlation in 2024"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.process_query(query)
        print(f"Confidence: {result['confidence']:.0%}")
        print(f"Sources: {len(result['sources'])}")
        print(f"Answer: {result['answer'][:200]}...")

if __name__ == "__main__":
    test_rag()