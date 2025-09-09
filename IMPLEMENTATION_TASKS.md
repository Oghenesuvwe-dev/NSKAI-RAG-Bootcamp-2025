# 🚀 Multi-Hop RAG Implementation Tasks

## 📋 **Task 1: Core Multi-Hop Engine** 
**Priority**: HIGH | **Time**: 6 hours

### **Implementation Steps**

#### **Step 1.1: Query Decomposition (2 hours)**
```python
# File: core/multi_hop_engine.py
class MultiHopRAG:
    def decompose_query(self, query: str) -> List[str]:
        prompt = f"""
        Break this complex query into 2-4 specific sub-questions:
        "{query}"
        
        Return JSON array: ["sub-question 1", "sub-question 2", "sub-question 3"]
        """
        # Use Groq LLM to decompose
        # Return list of sub-questions
```

#### **Step 1.2: Source Routing (2 hours)**
```python
def route_subquery(self, subquery: str) -> str:
    # Determine data source based on query content
    if any(word in subquery.lower() for word in ["stock", "price", "market"]):
        return "finance"
    elif any(word in subquery.lower() for word in ["premier", "nba", "sport"]):
        return "sports"
    elif any(word in subquery.lower() for word in ["business", "company", "revenue"]):
        return "business"
    return "general"
```

#### **Step 1.3: Evidence Collection (1 hour)**
```python
def collect_evidence(self, sub_questions: List[str]) -> List[Dict]:
    evidence = []
    for sq in sub_questions:
        source = self.route_subquery(sq)
        data = self.data_sources.get_data(sq, source)
        evidence.append({"question": sq, "source": source, "data": data})
    return evidence
```

#### **Step 1.4: Answer Synthesis (1 hour)**
```python
def synthesize_answer(self, query: str, evidence: List[Dict]) -> Dict:
    synthesis_prompt = f"""
    Answer: {query}
    
    Evidence: {json.dumps(evidence)}
    
    Provide clear, direct answer with sources cited.
    """
    # Use Groq LLM to synthesize final answer
```

---

## 📋 **Task 2: Live Data Integration**
**Priority**: HIGH | **Time**: 4 hours

### **Implementation Steps**

#### **Step 2.1: Yahoo Finance Integration (1.5 hours)**
```python
# File: core/data_sources.py
import yfinance as yf

class DataSources:
    def get_stock_data(self, symbol: str) -> Dict:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        info = ticker.info
        
        return {
            "symbol": symbol,
            "current_price": hist['Close'].iloc[-1],
            "change_pct": ((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100,
            "market_cap": info.get('marketCap'),
            "company_name": info.get('longName')
        }
```

#### **Step 2.2: TheSportsDB Integration (1.5 hours)**
```python
def get_sports_data(self, query: str) -> Dict:
    if "premier" in query.lower():
        url = "https://www.thesportsdb.com/api/v1/json/3/lookuptable.php?l=4328&s=2024-2025"
        response = requests.get(url)
        data = response.json()
        
        return {
            "competition": "Premier League",
            "standings": data.get('table', [])[:6],  # Top 6 teams
            "last_updated": datetime.now().isoformat()
        }
    
    elif "nba" in query.lower():
        # Similar implementation for NBA
        pass
```

#### **Step 2.3: Business Intelligence Sources (1 hour)**
```python
def get_business_data(self, query: str) -> Dict:
    # Market analysis and business intelligence
    business_topics = {
        "ai market": {"size": "$148B by 2030", "growth": "36.8% CAGR"},
        "microsoft openai": {"investment": "$13B", "impact": "Azure growth 35%"},
        "tech valuations": {"sector": "Technology", "trend": "AI-driven growth"}
    }
    
    for topic, data in business_topics.items():
        if topic in query.lower():
            return {"topic": topic, "analysis": data}
    
    return {"topic": "general business", "analysis": "Market trends and analysis"}
```

---

## 📋 **Task 3: Simple Clean UI**
**Priority**: MEDIUM | **Time**: 3 hours

### **Implementation Steps**

#### **Step 3.1: Basic Streamlit Interface (1 hour)**
```python
# File: app.py
import streamlit as st
from core.multi_hop_engine import MultiHopRAG

st.set_page_config(page_title="Multi-Hop RAG", page_icon="🎯", layout="centered")

st.title("🎯 Multi-Hop RAG Agent")
st.write("Ask complex questions about Sports, Business, and Finance")

# Initialize RAG engine
@st.cache_resource
def init_rag():
    return MultiHopRAG()

rag = init_rag()
```

#### **Step 3.2: Query Interface (1 hour)**
```python
# Example queries
examples = [
    "How did Liverpool's recent performance affect their Champions League chances vs Arsenal?",
    "Compare Microsoft's AI investment impact on stock vs Google's strategy",
    "What's the correlation between tech stocks and sports team valuations?"
]

selected = st.selectbox("Choose example or write custom:", ["Custom..."] + examples)
query = st.text_area("Your question:", value="" if selected == "Custom..." else selected)

if st.button("🔍 Analyze", type="primary"):
    if query.strip():
        with st.spinner("Processing multi-hop analysis..."):
            result = rag.process_query(query)
        
        # Display results
        st.success("✅ Analysis Complete")
        st.subheader("Answer")
        st.write(result["answer"])
```

#### **Step 3.3: Results Display (1 hour)**
```python
# Show reasoning process
with st.expander("🧠 Reasoning Process"):
    st.write("**Sub-questions analyzed:**")
    for i, sq in enumerate(result["sub_questions"], 1):
        st.write(f"{i}. {sq}")
    
    st.write("**Evidence gathered:**")
    for evidence in result["evidence"]:
        st.write(f"• **{evidence['source']}**: {evidence['data']}")
    
    st.write(f"**Sources**: {', '.join(result['sources'])}")
```

---

## 📋 **Task 4: Document Knowledge Base**
**Priority**: MEDIUM | **Time**: 4 hours

### **Implementation Steps**

#### **Step 4.1: ChromaDB Setup (1 hour)**
```python
# File: core/knowledge_base.py
import chromadb
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("knowledge")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
```

#### **Step 4.2: Document Ingestion (2 hours)**
```python
def load_documents(self):
    documents = {
        "sports_rules": "Premier League has 20 teams, top 4 qualify for Champions League...",
        "finance_basics": "Stock prices reflect company valuation, market cap = shares × price...",
        "business_analysis": "AI market growing 36.8% CAGR, Microsoft invested $13B in OpenAI..."
    }
    
    for doc_id, content in documents.items():
        embedding = self.embedder.encode(content)
        self.collection.add(
            documents=[content],
            embeddings=[embedding.tolist()],
            ids=[doc_id]
        )
```

#### **Step 4.3: Knowledge Retrieval (1 hour)**
```python
def query_knowledge(self, question: str, n_results: int = 2) -> List[str]:
    query_embedding = self.embedder.encode(question)
    results = self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []
```

---

## 📋 **Task 5: Answer Quality & Citations**
**Priority**: HIGH | **Time**: 3 hours

### **Implementation Steps**

#### **Step 5.1: Answer Formatting (1 hour)**
```python
# File: core/answer_formatter.py
class AnswerFormatter:
    def format_response(self, raw_answer: str, evidence: List[Dict]) -> Dict:
        # Clean up answer, remove code, ensure clarity
        clean_answer = self.clean_answer(raw_answer)
        sources = [e["source"] for e in evidence]
        confidence = self.calculate_confidence(evidence)
        
        return {
            "answer": clean_answer,
            "sources": list(set(sources)),
            "confidence": confidence,
            "evidence_count": len(evidence)
        }
```

#### **Step 5.2: Source Citations (1 hour)**
```python
def add_citations(self, answer: str, evidence: List[Dict]) -> str:
    # Add proper citations to answer
    cited_answer = answer
    for i, ev in enumerate(evidence, 1):
        source_name = ev["source"].title()
        cited_answer += f"\n\n**Source {i}**: {source_name} - {ev['data'][:100]}..."
    
    return cited_answer
```

#### **Step 5.3: Confidence Scoring (1 hour)**
```python
def calculate_confidence(self, evidence: List[Dict]) -> float:
    # Calculate confidence based on evidence quality
    base_confidence = 0.7
    
    # Boost for multiple sources
    if len(evidence) >= 3:
        base_confidence += 0.1
    
    # Boost for live data sources
    live_sources = ["finance", "sports"]
    if any(e["source"] in live_sources for e in evidence):
        base_confidence += 0.1
    
    return min(base_confidence, 0.95)
```

---

## 📋 **Task 6: Integration & Testing**
**Priority**: HIGH | **Time**: 2 hours

### **Implementation Steps**

#### **Step 6.1: Component Integration (1 hour)**
```python
# File: core/multi_hop_engine.py (complete integration)
class MultiHopRAG:
    def __init__(self):
        self.data_sources = DataSources()
        self.knowledge_base = KnowledgeBase()
        self.formatter = AnswerFormatter()
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    def process_query(self, query: str) -> Dict:
        # 1. Decompose query
        sub_questions = self.decompose_query(query)
        
        # 2. Collect evidence
        evidence = self.collect_evidence(sub_questions)
        
        # 3. Add knowledge context
        context = self.knowledge_base.query_knowledge(query)
        
        # 4. Synthesize answer
        raw_answer = self.synthesize_answer(query, evidence, context)
        
        # 5. Format response
        return self.formatter.format_response(raw_answer, evidence)
```

#### **Step 6.2: Testing & Validation (1 hour)**
```python
# Test cases
test_queries = [
    "How did Liverpool's performance affect Champions League chances vs Arsenal?",
    "Microsoft AI investment impact vs Google strategy on stock prices",
    "Tech stock and sports team valuation correlation in 2024"
]

for query in test_queries:
    result = rag.process_query(query)
    assert result["confidence"] > 0.6
    assert len(result["sources"]) >= 2
    assert "answer" in result
```

---

## 📦 **Updated Requirements**

```txt
# requirements.txt
streamlit>=1.28.0
groq>=0.4.1
yfinance>=0.2.18
requests>=2.31.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
python-dotenv>=1.0.0
```

---

## 🎯 **Success Metrics**

### **Functional Tests**
- [ ] Query decomposition works for complex questions
- [ ] Live data retrieval from Yahoo Finance & TheSportsDB
- [ ] Multi-hop reasoning across Sports/Finance/Business
- [ ] Clear answers with proper citations
- [ ] Simple, clean UI interface

### **Performance Targets**
- **Response Time**: <10 seconds
- **Accuracy**: Clear, factual answers
- **Sources**: 2+ data sources per complex query
- **Confidence**: >70% average confidence score

### **Example Success Cases**
- ✅ Sports: "Liverpool vs Arsenal Champions League analysis"
- ✅ Finance: "Microsoft vs Google AI stock impact"
- ✅ Cross-domain: "Tech stocks vs sports valuations"

**Total Implementation Time: 20 hours for complete multi-hop RAG system**