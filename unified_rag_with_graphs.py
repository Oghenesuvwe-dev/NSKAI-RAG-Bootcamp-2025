import streamlit as st
import json
import time
from datetime import datetime
import yfinance as yf
import requests
import os
from groq import Groq
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any

# Single file approach - no separate backend
st.set_page_config(
    page_title="Advanced Multi-Hop RAG Agent", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq client
@st.cache_resource
def init_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY") if hasattr(st, 'secrets') else os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Add to .streamlit/secrets.toml or environment")
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
.success-card {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚀 Advanced Multi-Hop RAG Agent</h1>
    <p>Finance • Sports • Business Intelligence | With Visualizations</p>
</div>
""", unsafe_allow_html=True)

# ==================== EMBEDDED BACKEND FUNCTIONS ====================

@st.cache_data(ttl=300)
def get_stock_data(symbol: str) -> Dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        info = ticker.info
        
        if hist.empty:
            return {"error": f"No data found for {symbol}"}
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        return {
            "symbol": symbol,
            "current_price": round(latest['Close'], 2),
            "previous_close": round(prev['Close'], 2),
            "change": round(latest['Close'] - prev['Close'], 2),
            "change_percent": round(((latest['Close'] - prev['Close']) / prev['Close']) * 100, 2),
            "volume": int(latest['Volume']),
            "market_cap": info.get('marketCap', 'N/A'),
            "company_name": info.get('longName', symbol),
            "sector": info.get('sector', 'N/A'),
            "source": "Yahoo Finance",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for {symbol}: {str(e)}"}

def get_football_data(query: str) -> Dict[str, Any]:
    football_db = {
        "premier_league": {
            "current_standings": [
                {"team": "Arsenal", "points": 84, "position": 1, "form": "WWWWW", "title_odds": "65%"},
                {"team": "Manchester City", "points": 82, "position": 2, "form": "WWLWW", "title_odds": "30%"},
                {"team": "Liverpool", "points": 78, "position": 3, "form": "WDWWW", "title_odds": "5%"}
            ],
            "prediction": "Arsenal leads by 2 points with superior form. 65% probability to win Premier League title.",
            "key_factors": ["Goal difference advantage", "Home form", "Injury list", "Remaining fixtures difficulty"]
        },
        "world_cup": {
            "top_contenders": [
                {"team": "Brazil", "fifa_ranking": 1, "win_probability": "28%", "key_strength": "Squad depth"},
                {"team": "Argentina", "fifa_ranking": 2, "win_probability": "24%", "key_strength": "Messi factor"},
                {"team": "France", "fifa_ranking": 3, "win_probability": "18%", "key_strength": "Experience"},
                {"team": "England", "fifa_ranking": 4, "win_probability": "15%", "key_strength": "Young talent"}
            ],
            "prediction": "Brazil remains favorite with 28% chance based on FIFA rankings and squad depth.",
            "analysis": "Historical performance and current form favor South American teams"
        }
    }
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["premier", "league", "arsenal", "city", "liverpool"]):
        return {
            "competition": "Premier League",
            "data": football_db["premier_league"],
            "source": "Football Analytics",
            "timestamp": datetime.now().isoformat()
        }
    elif any(word in query_lower for word in ["world", "cup", "brazil", "argentina", "fifa"]):
        return {
            "competition": "FIFA World Cup",
            "data": football_db["world_cup"],
            "source": "FIFA Rankings & Analytics",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "competition": "General Football",
        "data": {"message": "No specific competition data found"},
        "source": "Football Database",
        "timestamp": datetime.now().isoformat()
    }

def get_business_intelligence(query: str) -> Dict[str, Any]:
    business_db = {
        "genai_market": {
            "market_size": "$15.7B in 2024, projected $148.4B by 2030",
            "growth_rate": "36.8% CAGR",
            "key_players": [
                {"company": "OpenAI", "valuation": "$86B", "market_share": "25%"},
                {"company": "Google", "investment": "$70B", "market_share": "20%"},
                {"company": "Microsoft", "investment": "$13B in OpenAI", "market_share": "18%"}
            ],
            "impact_analysis": "GenAI driving 30% productivity gains in software development, 25% in content creation"
        },
        "microsoft_strategy": {
            "openai_partnership": {
                "investment": "$13B total investment",
                "azure_integration": "Exclusive cloud provider for OpenAI",
                "revenue_impact": "35% Azure growth attributed to AI services",
                "stock_performance": "+28% since partnership announcement"
            },
            "competitive_advantage": "First-mover advantage in enterprise AI integration"
        }
    }
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["genai", "artificial intelligence", "ai market"]):
        return {
            "topic": "GenAI Market Analysis",
            "data": business_db["genai_market"],
            "source": "Market Research & Analytics",
            "timestamp": datetime.now().isoformat()
        }
    elif any(word in query_lower for word in ["microsoft", "openai", "azure"]):
        return {
            "topic": "Microsoft AI Strategy",
            "data": business_db["microsoft_strategy"],
            "source": "Financial Analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "topic": "General Business Intelligence",
        "data": {"message": "AI transformation affecting all major tech companies"},
        "source": "Business Intelligence",
        "timestamp": datetime.now().isoformat()
    }

def intelligent_data_retrieval(sub_query: str, main_query: str) -> Dict[str, Any]:
    combined_query = (sub_query + " " + main_query).lower()
    
    if any(word in combined_query for word in ["football", "soccer", "premier", "league", "world", "cup"]):
        return get_football_data(combined_query)
    
    if any(word in combined_query for word in ["genai", "artificial", "intelligence", "openai", "microsoft", "google"]):
        return get_business_intelligence(combined_query)
    
    stock_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA", "AMZN", "META"]
    found_symbols = [s for s in stock_symbols if s.lower() in combined_query]
    
    if found_symbols or any(word in combined_query for word in ["stock", "price", "market", "earnings"]):
        symbol = found_symbols[0] if found_symbols else "AAPL"
        return get_stock_data(symbol)
    
    return {
        "topic": "General Analysis",
        "data": {"message": "General market and technology analysis"},
        "source": "Market Intelligence",
        "timestamp": datetime.now().isoformat()
    }

def process_multi_hop_query(query: str) -> Dict[str, Any]:
    if not client:
        return {"error": "Groq client not initialized. Please check API key."}
    
    try:
        start_time = time.time()
        
        # Query decomposition
        decomposition_prompt = f"""
        Decompose this complex query into 2-4 specific sub-questions for multi-hop reasoning:
        Query: "{query}"
        
        Return ONLY a JSON array of sub-questions:
        ["sub-question 1", "sub-question 2", "sub-question 3"]
        """
        
        decomp_response = client.chat.completions.create(
            messages=[{"role": "user", "content": decomposition_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [query]
        
        # Multi-hop data retrieval
        retrieved_data = []
        reasoning_steps = []
        all_sources = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            data_result = intelligent_data_retrieval(sub_q, query)
            
            if "error" not in data_result:
                retrieved_data.append({
                    "sub_question": sub_q,
                    "data_source": data_result.get("source", "Unknown"),
                    "content": json.dumps(data_result.get("data", {}), indent=2)[:500] + "...",
                    "full_data": data_result
                })
                
                source = data_result.get("source", "Unknown")
                all_sources.append(source)
                reasoning_steps.append(f"Step {i}: {sub_q} → Retrieved from {source}")
        
        # Evidence synthesis
        synthesis_prompt = f"""
        You are an expert analyst. Synthesize information from multiple sources to answer this query:
        
        Original Query: {query}
        
        Retrieved Evidence:
        {json.dumps([{
            "sub_question": item["sub_question"],
            "source": item["data_source"],
            "data": item["full_data"].get("data", {})
        } for item in retrieved_data], indent=2)}
        
        Provide a comprehensive analysis with specific facts, numbers, and proper citations.
        """
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        processing_time = time.time() - start_time
        unique_sources = list(set(all_sources))
        
        return {
            "answer": final_response.choices[0].message.content,
            "sub_questions": sub_questions,
            "reasoning_steps": reasoning_steps,
            "sources": unique_sources,
            "retrieved_data": retrieved_data,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# ==================== STREAMLIT UI ====================

# Sidebar
with st.sidebar:
    st.header("🎯 Analysis Categories")
    
    category = st.selectbox(
        "Choose Analysis Type:",
        ["💰 Finance & Markets", "⚽ Sports & Competitions", "🏢 Business Intelligence"]
    )
    
    st.divider()
    
    # System status
    st.header("🔧 System Status")
    if client:
        st.markdown("""
        <div class="success-card">
            <h4>✅ System Online</h4>
            <p>🤖 Groq LLM Ready</p>
            <p>📈 Real-Time Data Active</p>
            <p>📊 Visualizations Enabled</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ System Offline - Check API Key")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{category} Analysis Interface")
    
    # Category-specific example queries
    if category == "💰 Finance & Markets":
        example_queries = {
            "Select example...": "",
            "Microsoft OpenAI Impact": "How did Microsoft's $13B OpenAI investment affect Google's stock price and competitive position?",
            "GenAI Market Revolution": "What's the impact of generative AI adoption on NVIDIA, Microsoft, and tech valuations?",
            "Custom Analysis": ""
        }
    elif category == "⚽ Sports & Competitions":
        example_queries = {
            "Select example...": "",
            "Premier League Title Race": "Based on current standings and form, who will win the Premier League?",
            "World Cup 2026 Predictions": "Which team has the best chance to win the 2026 FIFA World Cup?",
            "Custom Analysis": ""
        }
    else:  # Business Intelligence
        example_queries = {
            "Select example...": "",
            "GenAI Economic Impact": "How is the $148B generative AI market transforming tech industry economics?",
            "Big Tech AI Race": "Compare Microsoft, Google, and Amazon's AI strategies and market impact",
            "Custom Analysis": ""
        }
    
    selected_query = st.selectbox("Choose analysis query:", list(example_queries.keys()))
    
    if selected_query == "Custom Analysis":
        query = st.text_area(
            "Enter your multi-hop analysis question:", 
            height=120,
            placeholder="Ask complex questions requiring analysis across multiple data sources..."
        )
    else:
        query = example_queries[selected_query]
        if query:
            st.text_area("Selected Query:", value=query, height=120, disabled=True)

with col2:
    st.subheader("📊 Quick Data Lookup")
    
    symbol = st.text_input("Stock Symbol:", placeholder="AAPL, MSFT, GOOGL...")
    
    if st.button("📈 Get Stock Data", use_container_width=True) and symbol:
        with st.spinner("Fetching data..."):
            stock_data = get_stock_data(symbol.upper())
            
            if "error" not in stock_data:
                st.success(f"**{stock_data['company_name']}** ({stock_data['symbol']})")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Price", f"${stock_data['current_price']}")
                with col_b:
                    st.metric("Change", f"{stock_data['change']}", f"{stock_data['change_percent']}%")
            else:
                st.error(stock_data['error'])

# Execute analysis
if st.button("🚀 Execute Multi-Hop Analysis", type="primary", use_container_width=True):
    if query and query.strip():
        with st.spinner("🧠 Processing multi-hop analysis..."):
            result = process_multi_hop_query(query.strip())
            
            if "error" not in result:
                # Success metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("⏱️ Processing Time", f"{result['processing_time']:.2f}s")
                with col_metric2:
                    st.metric("📈 Sources Used", len(result['sources']))
                with col_metric3:
                    st.metric("🔍 Sub-Questions", len(result['sub_questions']))
                
                st.success(f"✅ {category} Analysis Complete!")
                
                # Main analysis result
                st.subheader("💡 Multi-Hop Analysis Result")
                st.markdown(f"""
                <div class="category-card">
                    {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Sub-questions and reasoning
                col_reasoning1, col_reasoning2 = st.columns(2)
                
                with col_reasoning1:
                    st.subheader("❓ Sub-Questions")
                    for i, sq in enumerate(result['sub_questions'], 1):
                        st.text(f"{i}. {sq}")
                
                with col_reasoning2:
                    st.subheader("🔄 Reasoning Steps")
                    for step in result['reasoning_steps']:
                        st.text(f"• {step}")
                
                # Sources
                st.subheader("📡 Data Sources")
                source_cols = st.columns(min(len(result['sources']), 4))
                for i, source in enumerate(result['sources']):
                    with source_cols[i % 4]:
                        st.success(f"📈 {source}")
                
                # Detailed evidence
                with st.expander("🔍 Detailed Evidence & Data"):
                    for i, item in enumerate(result['retrieved_data'], 1):
                        st.markdown(f"**Evidence {i}: {item['sub_question']}**")
                        st.info(f"Source: {item['data_source']}")
                        st.code(item['content'], language='json')
                        st.divider()
                
                # 📥 Export Analysis & Visualizations - GRAPHS DISPLAY HERE
                st.subheader("📥 Export Analysis & Visualizations")
                
                # Create visualizations
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    # Processing time gauge
                    fig_time = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['processing_time'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Processing Time (seconds)"},
                        gauge = {
                            'axis': {'range': [None, 10]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 2], 'color': "lightgray"},
                                {'range': [2, 5], 'color': "gray"}],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 8}}
                    ))
                    fig_time.update_layout(height=300)
                    st.plotly_chart(fig_time, use_container_width=True)
                
                with col_viz2:
                    # Sources pie chart
                    if len(result['sources']) > 1:
                        fig_sources = px.pie(
                            values=[1] * len(result['sources']),
                            names=result['sources'],
                            title="Data Sources Distribution"
                        )
                        fig_sources.update_layout(height=300)
                        st.plotly_chart(fig_sources, use_container_width=True)
                    else:
                        st.info(f"📈 Single Source: {result['sources'][0]}")
                
                # Sub-questions complexity chart
                if len(result['sub_questions']) > 1:
                    fig_questions = go.Figure(data=[
                        go.Bar(
                            x=[f"Q{i+1}" for i in range(len(result['sub_questions']))],
                            y=[len(sq.split()) for sq in result['sub_questions']],
                            text=[sq[:30] + "..." if len(sq) > 30 else sq for sq in result['sub_questions']],
                            textposition='outside',
                            marker_color='#764ba2'
                        )
                    ])
                    fig_questions.update_layout(
                        title="Sub-Questions Analysis (Word Complexity)",
                        xaxis_title="Sub-Questions",
                        yaxis_title="Word Count",
                        height=400
                    )
                    st.plotly_chart(fig_questions, use_container_width=True)
                
                # Download options
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    st.download_button(
                        "📄 Download JSON",
                        json.dumps(result, indent=2),
                        f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                
                with col_export2:
                    report = f"""# {category} Analysis Report

**Query:** {query}
**Timestamp:** {result['timestamp']}
**Processing Time:** {result['processing_time']:.2f}s

## Analysis Result
{result['answer']}

## Sub-Questions Analyzed
{chr(10).join([f"{i}. {sq}" for i, sq in enumerate(result['sub_questions'], 1)])}

## Data Sources
{', '.join(result['sources'])}

## Reasoning Chain
{chr(10).join(result['reasoning_steps'])}
"""
                    st.download_button(
                        "📊 Download Report",
                        report,
                        f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown"
                    )
                
                with col_export3:
                    summary = f"""Analysis Summary:
- Query: {query[:50]}...
- Processing Time: {result['processing_time']:.2f}s
- Sources: {len(result['sources'])}
- Sub-Questions: {len(result['sub_questions'])}
- Category: {category}
"""
                    st.download_button(
                        "📋 Download Summary",
                        summary,
                        f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        "text/plain"
                    )
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
    else:
        st.warning("⚠️ Please select or enter a query to analyze")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 <strong>Advanced Multi-Hop RAG Agent</strong> | With Interactive Visualizations</p>
    <p>Finance • Sports • Business Intelligence | Real-World Data Analysis | NSKAI Bootcamp 2025</p>
</div>
""", unsafe_allow_html=True)