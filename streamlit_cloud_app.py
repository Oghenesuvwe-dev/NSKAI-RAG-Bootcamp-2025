import streamlit as st
import json
import time
from datetime import datetime
import yfinance as yf
import requests
import os
from groq import Groq

# Streamlit Cloud compatible version - no separate backend needed
st.set_page_config(
    page_title="Advanced Multi-Hop RAG Agent", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY not found in secrets or environment")
        return None
    return Groq(api_key=api_key)

client = init_groq_client()

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.category-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Multi-Hop RAG Agent</h1>
    <p>Finance • Sports • Business Intelligence | Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)

# Embedded data functions (no external API needed)
def get_stock_data(symbol: str) -> dict:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1mo")
        info = ticker.info
        
        if hist.empty:
            return {"error": f"No data found for {symbol}"}
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        return {
            "symbol": symbol,
            "current_price": round(latest['Close'], 2),
            "change": round(latest['Close'] - prev['Close'], 2),
            "change_percent": round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'N/A'),
            "source": "Yahoo Finance"
        }
    except Exception as e:
        return {"error": str(e)}

def get_football_data(query: str) -> dict:
    football_db = {
        "premier_league": {
            "standings": [
                {"team": "Arsenal", "points": 84, "position": 1},
                {"team": "Manchester City", "points": 82, "position": 2},
                {"team": "Liverpool", "points": 78, "position": 3}
            ],
            "prediction": "Arsenal leads by 2 points. 65% chance to win title."
        },
        "world_cup": {
            "favorites": ["Brazil", "Argentina", "France", "England"],
            "prediction": "Brazil 28% chance, Argentina 24% based on FIFA rankings."
        }
    }
    
    if "premier" in query.lower():
        return {"data": football_db["premier_league"], "source": "Football Data"}
    elif "world" in query.lower():
        return {"data": football_db["world_cup"], "source": "FIFA Rankings"}
    
    return {"data": {"message": "General football data"}, "source": "Football DB"}

def get_business_intelligence(query: str) -> dict:
    business_db = {
        "genai_impact": {
            "market_size": "$15.7B in 2024, projected $148B by 2030",
            "leaders": ["OpenAI", "Google", "Microsoft", "Anthropic"],
            "impact": "30% productivity gains in software development"
        },
        "microsoft_deals": {
            "openai_investment": "$13B investment in OpenAI",
            "azure_growth": "35% revenue increase from AI services",
            "market_impact": "Stock up 28% since partnership"
        }
    }
    
    if "genai" in query.lower() or "ai impact" in query.lower():
        return {"data": business_db["genai_impact"], "source": "Market Research"}
    elif "microsoft" in query.lower() or "openai" in query.lower():
        return {"data": business_db["microsoft_deals"], "source": "Financial Reports"}
    
    return {"data": {"message": "General business intelligence"}, "source": "Business Analysis"}

def process_query(query: str):
    if not client:
        return {"error": "Groq client not initialized"}
    
    try:
        # Query decomposition
        decomp_response = client.chat.completions.create(
            messages=[{
                "role": "user", 
                "content": f'Break down into 2-3 sub-questions: "{query}". Return JSON array only.'
            }],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [query]
        
        # Retrieve data for each sub-question
        retrieved_data = []
        sources = []
        
        for sub_q in sub_questions:
            query_lower = (sub_q + " " + query).lower()
            
            # Route to appropriate data source
            if any(word in query_lower for word in ["stock", "price", "market", "aapl", "msft", "googl", "nvda"]):
                symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN"]
                found_symbol = next((s for s in symbols if s.lower() in query_lower), "AAPL")
                data = get_stock_data(found_symbol)
                if "error" not in data:
                    retrieved_data.append({
                        "sub_question": sub_q,
                        "data": data,
                        "source": "Yahoo Finance"
                    })
                    sources.append("Yahoo Finance")
            
            elif any(word in query_lower for word in ["football", "premier", "world", "cup"]):
                data = get_football_data(query_lower)
                retrieved_data.append({
                    "sub_question": sub_q,
                    "data": data["data"],
                    "source": data["source"]
                })
                sources.append(data["source"])
            
            elif any(word in query_lower for word in ["genai", "microsoft", "openai", "business"]):
                data = get_business_intelligence(query_lower)
                retrieved_data.append({
                    "sub_question": sub_q,
                    "data": data["data"],
                    "source": data["source"]
                })
                sources.append(data["source"])
        
        # Synthesize answer
        synthesis_prompt = f"""Query: {query}

Evidence: {json.dumps(retrieved_data, indent=2)}

Provide a comprehensive answer using the evidence."""
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1200
        )
        
        return {
            "answer": final_response.choices[0].message.content,
            "sub_questions": sub_questions,
            "retrieved_data": retrieved_data,
            "sources": list(set(sources))
        }
        
    except Exception as e:
        return {"error": str(e)}

# Sidebar
with st.sidebar:
    st.header("🎯 Query Categories")
    
    category = st.selectbox(
        "Choose Analysis Type:",
        ["💰 Finance & Markets", "⚽ Sports & Competitions", "🏢 Business Intelligence"]
    )
    
    st.success("✅ Streamlit Cloud Ready")
    st.info("🤖 Groq LLM: llama-3.3-70b")
    st.info("📈 Real-Time: Yahoo Finance")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{category} Query Interface")
    
    # Category queries
    if category == "💰 Finance & Markets":
        queries = {
            "Select example...": "",
            "Microsoft AI Impact": "How did Microsoft's OpenAI investment affect Google's stock price?",
            "Tech Stock Analysis": "Compare NVIDIA and Microsoft stock performance in the AI era",
            "Custom Query": ""
        }
    elif category == "⚽ Sports & Competitions":
        queries = {
            "Select example...": "",
            "Premier League": "Who will win the Premier League based on current standings?",
            "World Cup": "Which team has the best chance in the next World Cup?",
            "Custom Query": ""
        }
    else:
        queries = {
            "Select example...": "",
            "GenAI Revolution": "How is generative AI transforming tech industry economics?",
            "Big Tech AI Race": "Compare Microsoft, Google AI strategies and market impact",
            "Custom Query": ""
        }
    
    selected = st.selectbox("Choose query:", list(queries.keys()))
    
    if selected == "Custom Query":
        query = st.text_area("Enter your question:", height=120)
    else:
        query = queries[selected]
        if query:
            st.text_area("Selected Query:", value=query, height=120, disabled=True)

with col2:
    st.subheader("📊 Quick Stock Check")
    symbol = st.text_input("Symbol:", placeholder="AAPL")
    
    if st.button("Get Data") and symbol:
        data = get_stock_data(symbol.upper())
        if "error" not in data:
            st.metric("Price", f"${data['current_price']}", f"{data['change']} ({data['change_percent']}%)")
            st.write(f"Company: {data['company_name']}")
        else:
            st.error(data['error'])

# Execute analysis
if st.button("🚀 Execute Analysis", type="primary") and query:
    with st.spinner("Processing..."):
        start_time = time.time()
        result = process_query(query)
        processing_time = time.time() - start_time
        
        if "error" not in result:
            st.success(f"✅ Analysis complete! ({processing_time:.2f}s)")
            
            # Results
            st.subheader("💡 Analysis Result")
            st.markdown(f"""
            <div class="category-card">
                {result['answer']}
            </div>
            """, unsafe_allow_html=True)
            
            # Sub-questions
            if result.get("sub_questions"):
                st.subheader("🔄 Sub-Questions")
                for i, sq in enumerate(result["sub_questions"], 1):
                    st.text(f"{i}. {sq}")
            
            # Sources
            if result.get("sources"):
                st.subheader("📡 Sources")
                for source in result["sources"]:
                    st.success(f"📈 {source}")
            
            # Export
            st.download_button(
                "📄 Download Results",
                json.dumps(result, indent=2),
                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
        else:
            st.error(f"Error: {result['error']}")

st.divider()
st.markdown("**🚀 Advanced Multi-Hop RAG Agent** | Streamlit Cloud | NSKAI Bootcamp 2025")