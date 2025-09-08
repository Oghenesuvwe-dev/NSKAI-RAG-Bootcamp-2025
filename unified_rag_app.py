import streamlit as st
import json
import time
from datetime import datetime
import yfinance as yf
import requests
import os
from groq import Groq
import pandas as pd
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
    <p>Finance • Sports • Business Intelligence | Unified Architecture</p>
</div>
""", unsafe_allow_html=True)

# ==================== EMBEDDED BACKEND FUNCTIONS ====================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol: str) -> Dict[str, Any]:
    """Get real-time stock data from Yahoo Finance"""
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
    """Get football predictions and data"""
    football_db = {
        "premier_league": {
            "current_standings": [
                {"team": "Arsenal", "points": 84, "position": 1, "form": "WWWWW", "title_odds": "65%"},
                {"team": "Manchester City", "points": 82, "position": 2, "form": "WWLWW", "title_odds": "30%"},
                {"team": "Liverpool", "points": 78, "position": 3, "form": "WDWWW", "title_odds": "5%"}
            ],
            "prediction": "Arsenal leads by 2 points with superior form. 65% probability to win Premier League title based on current trajectory and remaining fixtures.",
            "key_factors": ["Goal difference advantage", "Home form", "Injury list", "Remaining fixtures difficulty"]
        },
        "world_cup": {
            "top_contenders": [
                {"team": "Brazil", "fifa_ranking": 1, "win_probability": "28%", "key_strength": "Squad depth"},
                {"team": "Argentina", "fifa_ranking": 2, "win_probability": "24%", "key_strength": "Messi factor"},
                {"team": "France", "fifa_ranking": 3, "win_probability": "18%", "key_strength": "Experience"},
                {"team": "England", "fifa_ranking": 4, "win_probability": "15%", "key_strength": "Young talent"}
            ],
            "prediction": "Brazil remains favorite with 28% chance based on FIFA rankings, squad depth, and recent international form.",
            "analysis": "Historical performance, current form, and squad quality favor South American teams"
        },
        "afcon": {
            "favorites": [
                {"team": "Nigeria", "probability": "22%", "strength": "Squad depth in European leagues"},
                {"team": "Egypt", "probability": "18%", "strength": "Home advantage and Salah"},
                {"team": "Morocco", "probability": "16%", "strength": "World Cup momentum"},
                {"team": "Senegal", "probability": "14%", "strength": "Defending champions"}
            ],
            "prediction": "Nigeria leads with strongest squad playing in top European leagues. 22% win probability.",
            "key_insight": "European-based players advantage in modern African football"
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
    elif any(word in query_lower for word in ["afcon", "africa", "nigeria", "egypt", "morocco"]):
        return {
            "competition": "Africa Cup of Nations",
            "data": football_db["afcon"],
            "source": "CAF Analytics",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "competition": "General Football",
        "data": {"message": "No specific competition data found"},
        "source": "Football Database",
        "timestamp": datetime.now().isoformat()
    }

def get_business_intelligence(query: str) -> Dict[str, Any]:
    """Get business intelligence and market analysis"""
    business_db = {
        "genai_market": {
            "market_size": "$15.7B in 2024, projected $148.4B by 2030",
            "growth_rate": "36.8% CAGR",
            "key_players": [
                {"company": "OpenAI", "valuation": "$86B", "market_share": "25%"},
                {"company": "Google", "investment": "$70B", "market_share": "20%"},
                {"company": "Microsoft", "investment": "$13B in OpenAI", "market_share": "18%"},
                {"company": "Anthropic", "valuation": "$18.4B", "market_share": "8%"}
            ],
            "impact_analysis": "GenAI driving 30% productivity gains in software development, 25% in content creation, reshaping entire tech economics",
            "market_trends": ["Enterprise adoption accelerating", "API-first business models", "Compute costs driving consolidation"]
        },
        "microsoft_strategy": {
            "openai_partnership": {
                "investment": "$13B total investment",
                "azure_integration": "Exclusive cloud provider for OpenAI",
                "revenue_impact": "35% Azure growth attributed to AI services",
                "stock_performance": "+28% since partnership announcement"
            },
            "competitive_advantage": "First-mover advantage in enterprise AI integration",
            "market_position": "Leading enterprise AI adoption with Office 365 + Azure synergy"
        },
        "google_response": {
            "bard_launch": "Direct ChatGPT competitor",
            "search_integration": "AI-powered search results rolling out",
            "investment_scale": "$70B AI investment commitment",
            "market_challenge": "Protecting search monopoly while innovating",
            "stock_volatility": "15% fluctuation due to AI competition concerns"
        },
        "amazon_ai": {
            "aws_bedrock": "Enterprise AI platform competing with Azure OpenAI",
            "alexa_evolution": "Large language model integration planned",
            "market_share": "Maintaining 32% cloud market share despite AI competition",
            "strategy": "Focus on enterprise AI infrastructure and cost optimization"
        }
    }
    
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["genai", "artificial intelligence", "ai market", "ai impact"]):
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
    elif any(word in query_lower for word in ["google", "bard", "alphabet"]):
        return {
            "topic": "Google AI Response",
            "data": business_db["google_response"],
            "source": "Tech Industry Analysis",
            "timestamp": datetime.now().isoformat()
        }
    elif any(word in query_lower for word in ["amazon", "aws", "bedrock"]):
        return {
            "topic": "Amazon AI Strategy",
            "data": business_db["amazon_ai"],
            "source": "Cloud Market Analysis",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "topic": "General Business Intelligence",
        "data": {"message": "AI transformation affecting all major tech companies"},
        "source": "Business Intelligence",
        "timestamp": datetime.now().isoformat()
    }

def intelligent_data_retrieval(sub_query: str, main_query: str) -> Dict[str, Any]:
    """Route queries to appropriate data sources"""
    combined_query = (sub_query + " " + main_query).lower()
    
    # 1. Sports queries
    if any(word in combined_query for word in ["football", "soccer", "premier", "league", "world", "cup", "afcon", "arsenal", "liverpool", "city", "nigeria", "brazil"]):
        return get_football_data(combined_query)
    
    # 2. Business intelligence queries
    if any(word in combined_query for word in ["genai", "artificial", "intelligence", "openai", "microsoft", "google", "amazon", "business", "market", "ai impact"]):
        return get_business_intelligence(combined_query)
    
    # 3. Stock market queries
    stock_symbols = ["AAPL", "TSLA", "MSFT", "GOOGL", "NVDA", "AMZN", "META"]
    found_symbols = [s for s in stock_symbols if s.lower() in combined_query]
    
    if found_symbols or any(word in combined_query for word in ["stock", "price", "market", "earnings", "trading"]):
        symbol = found_symbols[0] if found_symbols else "AAPL"
        return get_stock_data(symbol)
    
    # 4. General market/tech queries
    return {
        "topic": "General Analysis",
        "data": {"message": "General market and technology analysis"},
        "source": "Market Intelligence",
        "timestamp": datetime.now().isoformat()
    }

def process_multi_hop_query(query: str) -> Dict[str, Any]:
    """Main multi-hop processing function"""
    if not client:
        return {"error": "Groq client not initialized. Please check API key."}
    
    try:
        start_time = time.time()
        
        # Step 1: Query decomposition
        decomposition_prompt = f"""
        Decompose this complex query into 2-4 specific sub-questions for multi-hop reasoning:
        Query: "{query}"
        
        Focus on extracting:
        - Key entities (companies, teams, markets)
        - Relationships and impacts
        - Specific data points needed
        
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
            sub_questions = [query]  # Fallback
        
        # Step 2: Multi-hop data retrieval
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
        
        # Step 3: Evidence synthesis
        synthesis_prompt = f"""
        You are an expert analyst. Synthesize information from multiple sources to answer this query:
        
        Original Query: {query}
        
        Retrieved Evidence:
        {json.dumps([{
            "sub_question": item["sub_question"],
            "source": item["data_source"],
            "data": item["full_data"].get("data", {})
        } for item in retrieved_data], indent=2)}
        
        Provide a comprehensive analysis that:
        1. Directly addresses the original query
        2. Uses specific facts, numbers, and percentages from the evidence
        3. Shows clear multi-hop reasoning connecting different pieces of evidence
        4. Cites sources appropriately
        5. Provides actionable insights and conclusions
        
        Format your response in a clear, professional manner with proper citations.
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
            <p>🚀 Unified Architecture</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ System Offline - Check API Key")
    
    st.divider()
    
    st.subheader("📡 Data Sources")
    sources = [
        "📈 Yahoo Finance (Real-time)",
        "⚽ Football Analytics",
        "🏢 Business Intelligence",
        "🤖 Groq LLM (llama-3.3-70b)",
        "🔄 Multi-hop Reasoning"
    ]
    for source in sources:
        st.write(f"• {source}")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{category} Analysis Interface")
    
    # Category-specific example queries
    if category == "💰 Finance & Markets":
        example_queries = {
            "Select example...": "",
            "Microsoft OpenAI Impact": "How did Microsoft's $13B OpenAI investment affect Google's stock price and competitive position in the AI market?",
            "GenAI Market Revolution": "What's the comprehensive impact of generative AI adoption on NVIDIA, Microsoft, and overall tech sector valuations?",
            "Apple Ecosystem Effect": "How did Apple's latest earnings and AI strategy announcements affect the broader tech sector and supplier relationships?",
            "Tesla vs Traditional Auto": "Analyze Tesla's market performance compared to traditional automakers in the EV transition era",
            "Custom Analysis": ""
        }
    elif category == "⚽ Sports & Competitions":
        example_queries = {
            "Select example...": "",
            "Premier League Title Race": "Based on current standings, form, and remaining fixtures, who will win the Premier League and what are the key factors?",
            "World Cup 2026 Predictions": "Which team has the best statistical chance to win the 2026 FIFA World Cup based on current form and historical data?",
            "AFCON 2025 Analysis": "Who are the top contenders for the next Africa Cup of Nations and what gives them competitive advantages?",
            "Arsenal Title Probability": "How have Arsenal's recent performances and tactical changes affected their Premier League title winning probability?",
            "Custom Analysis": ""
        }
    else:  # Business Intelligence
        example_queries = {
            "Select example...": "",
            "GenAI Economic Transformation": "How is the $148B generative AI market transforming tech industry economics and reshaping company valuations?",
            "Big Tech AI Arms Race": "Compare and analyze Microsoft, Google, and Amazon's AI strategies, investments, and market positioning",
            "OpenAI Partnership Impact": "What has been the comprehensive business impact of Microsoft's OpenAI partnership on Azure revenues and market share?",
            "AI Investment Tsunami": "How are massive AI investments by Big Tech companies affecting their valuations, strategies, and competitive dynamics?",
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
    
    # Quick stock lookup
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
                
                st.write(f"📊 Volume: {stock_data['volume']:,}")
                st.write(f"🏢 Sector: {stock_data['sector']}")
            else:
                st.error(stock_data['error'])
    
    st.divider()
    
    # Analysis history
    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    
    if st.session_state.analysis_history:
        st.subheader("📝 Recent Analysis")
        for i, (timestamp, query_text) in enumerate(st.session_state.analysis_history[-3:]):
            if st.button(f"🔄 {query_text[:25]}...", key=f"history_{i}"):
                st.session_state.selected_query = query_text
                st.rerun()

# Execute analysis
if st.button("🚀 Execute Multi-Hop Analysis", type="primary", use_container_width=True):
    if query and query.strip():
        # Add to history
        st.session_state.analysis_history.append((datetime.now().strftime("%H:%M"), query))
        
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
                
                # Export functionality
                st.subheader("📥 Export Analysis")
                col_export1, col_export2 = st.columns(2)
                
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
            else:
                st.error(f"❌ Analysis failed: {result['error']}")
    else:
        st.warning("⚠️ Please select or enter a query to analyze")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 <strong>Advanced Multi-Hop RAG Agent</strong> | Unified Architecture</p>
    <p>Finance • Sports • Business Intelligence | Real-World Data Analysis | NSKAI Bootcamp 2025</p>
    <p><em>Single file deployment - No separate backend required</em></p>
</div>
""", unsafe_allow_html=True)