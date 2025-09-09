"""
🎯 Simple Multi-Hop RAG Agent
Clean, focused interface for multi-hop reasoning
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
import yfinance as yf
from groq import Groq
import os

# Simple page config
st.set_page_config(
    page_title="Multi-Hop RAG Agent",
    page_icon="🎯",
    layout="centered"
)

# Initialize Groq client
@st.cache_resource
def init_groq():
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY required")
        st.stop()
    return Groq(api_key=api_key)

client = init_groq()

# Simple data sources
def get_stock_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return None
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        return {
            "symbol": symbol,
            "price": round(latest['Close'], 2),
            "change": round(latest['Close'] - prev['Close'], 2),
            "change_pct": round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2)
        }
    except:
        return None

def get_sports_data(query):
    # Simple sports data
    if "premier" in query.lower() or "football" in query.lower():
        return {
            "competition": "Premier League",
            "leader": "Liverpool",
            "points": 45,
            "analysis": "Liverpool leads with strong attacking form"
        }
    elif "nba" in query.lower() or "basketball" in query.lower():
        return {
            "competition": "NBA",
            "leader": "Oklahoma City Thunder",
            "wins": 33,
            "analysis": "Thunder leads Western Conference with young talent"
        }
    return None

# Core multi-hop RAG function
def multi_hop_rag(query):
    """Simple multi-hop reasoning"""
    
    # Step 1: Break down query
    decompose_prompt = f"""
    Break this query into 2-3 simple sub-questions:
    "{query}"
    
    Return only a JSON list: ["question1", "question2", "question3"]
    """
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": decompose_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200
        )
        
        sub_questions = json.loads(response.choices[0].message.content)
    except:
        sub_questions = [query]
    
    # Step 2: Get data for each sub-question
    evidence = []
    for sq in sub_questions:
        sq_lower = sq.lower()
        
        # Check for stock symbols
        stock_symbols = ["aapl", "msft", "googl", "nvda", "tsla", "amzn", "meta"]
        found_stock = None
        for symbol in stock_symbols:
            if symbol in sq_lower:
                found_stock = symbol.upper()
                break
        
        if found_stock:
            stock_data = get_stock_data(found_stock)
            if stock_data:
                evidence.append({
                    "question": sq,
                    "source": "Stock Market",
                    "data": f"{stock_data['symbol']}: ${stock_data['price']} ({stock_data['change']:+.2f}, {stock_data['change_pct']:+.1f}%)"
                })
        
        # Check for sports
        elif any(word in sq_lower for word in ["premier", "nba", "football", "basketball", "sport"]):
            sports_data = get_sports_data(sq)
            if sports_data:
                evidence.append({
                    "question": sq,
                    "source": "Sports Data",
                    "data": f"{sports_data['competition']}: {sports_data['leader']} leads with {sports_data.get('points', sports_data.get('wins', 'N/A'))} points/wins"
                })
        
        # General business/market data
        else:
            evidence.append({
                "question": sq,
                "source": "Market Analysis",
                "data": "Current market conditions and business trends analysis"
            })
    
    # Step 3: Synthesize answer
    synthesis_prompt = f"""
    Answer this question using the evidence provided:
    
    Question: {query}
    
    Evidence:
    {json.dumps(evidence, indent=2)}
    
    Provide a clear, direct answer in 2-3 sentences. Be specific and use the data provided.
    """
    
    try:
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300
        )
        
        answer = final_response.choices[0].message.content
    except Exception as e:
        answer = f"Error generating response: {str(e)}"
    
    return {
        "answer": answer,
        "sub_questions": sub_questions,
        "evidence": evidence,
        "sources": list(set([e["source"] for e in evidence]))
    }

# Simple UI
st.title("🎯 Multi-Hop RAG Agent")
st.write("Ask complex questions that require analysis across multiple data sources")

# Example queries
st.subheader("📝 Example Queries")
examples = [
    "How is Apple stock performing compared to Microsoft?",
    "Who is leading the Premier League and NBA this season?",
    "Compare Tesla stock with current sports market trends",
    "What's the relationship between tech stocks and sports valuations?"
]

selected_example = st.selectbox("Choose an example or write your own:", ["Custom query..."] + examples)

# Query input
if selected_example == "Custom query...":
    query = st.text_area("Your question:", placeholder="Ask a question that needs multiple data sources...")
else:
    query = st.text_area("Your question:", value=selected_example)

# Analyze button
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if query.strip():
        with st.spinner("🧠 Processing multi-hop analysis..."):
            start_time = time.time()
            result = multi_hop_rag(query.strip())
            processing_time = time.time() - start_time
        
        # Display results
        st.success(f"✅ Analysis complete ({processing_time:.1f}s)")
        
        # Main answer
        st.subheader("💡 Answer")
        st.write(result["answer"])
        
        # Show reasoning process
        with st.expander("🔍 How I reasoned through this"):
            st.write("**Sub-questions analyzed:**")
            for i, sq in enumerate(result["sub_questions"], 1):
                st.write(f"{i}. {sq}")
            
            st.write("**Evidence gathered:**")
            for evidence in result["evidence"]:
                st.write(f"• **{evidence['source']}**: {evidence['data']}")
            
            st.write(f"**Sources used**: {', '.join(result['sources'])}")
    else:
        st.warning("Please enter a question to analyze")

# Simple footer
st.divider()
st.caption("🚀 Multi-Hop RAG Agent - Analyzes complex questions using multiple data sources")