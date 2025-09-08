import streamlit as st
import requests
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="Advanced Multi-Hop RAG Agent", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
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
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    <p>Finance • Sports • Business Intelligence with Real-World Data</p>
</div>
""", unsafe_allow_html=True)

API_URL = "http://localhost:8004"

# Sidebar for categories and system status
with st.sidebar:
    st.header("🎯 Query Categories")
    
    category = st.selectbox(
        "Choose Analysis Type:",
        ["💰 Finance & Markets", "⚽ Sports & Competitions", "🏢 Business Intelligence"]
    )
    
    st.divider()
    
    # System status
    st.header("🔧 System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2)
        if health.status_code == 200:
            data = health.json()
            st.success("✅ Multi-Hop RAG Online")
            st.info(f"🤖 Model: {data.get('model', 'Unknown')}")
            st.info(f"📈 Real-Time Data: {data.get('real_time_data', False)}")
            
            st.subheader("📡 Enhanced Data Sources")
            sources = ["Yahoo Finance", "Football Data API", "Business Intelligence", "Market News API"]
            for source in sources:
                st.write(f"• {source}")
        else:
            st.error("❌ System Offline")
    except:
        st.error("❌ Cannot connect to system")
    
    st.divider()
    
    st.subheader("🎯 Supported Analysis")
    analysis_types = [
        "📈 Stock market analysis",
        "⚽ Football predictions", 
        "🏆 Sports competitions",
        "🤖 AI market impact",
        "💼 Business intelligence",
        "🔗 Multi-hop reasoning"
    ]
    for analysis in analysis_types:
        st.write(f"✓ {analysis}")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{category} Query Interface")
    
    # Category-specific queries
    if category == "💰 Finance & Markets":
        market_queries = {
            "Select example...": "",
            "Microsoft OpenAI Impact": "How did Microsoft's OpenAI investment affect Google's stock price and market position?",
            "GenAI Market Revolution": "What's the impact of generative AI adoption on NVIDIA, Microsoft, and tech valuations?",
            "Apple Earnings Ripple": "How did Apple's latest earnings affect the broader tech sector and supplier stocks?",
            "Tesla Market Dynamics": "Analyze Tesla's stock performance and its correlation with EV market trends",
            "Custom Query": ""
        }
    elif category == "⚽ Sports & Competitions":
        market_queries = {
            "Select example...": "",
            "Premier League Title Race": "Based on current form and fixtures, who will win the Premier League this season?",
            "World Cup Predictions": "Which team has the best statistical chance to win the next FIFA World Cup?",
            "AFCON Championship": "Who are the top favorites for the next Africa Cup of Nations tournament?",
            "Arsenal Title Chances": "How did Arsenal's recent performance affect their Premier League title probability?",
            "Custom Query": ""
        }
    else:  # Business Intelligence
        market_queries = {
            "Select example...": "",
            "GenAI Economic Impact": "How is generative AI transforming tech industry economics and market valuations?",
            "Big Tech AI Arms Race": "Compare Microsoft, Google, and Amazon's AI strategies and their market impact",
            "OpenAI Partnership Analysis": "Analyze the business impact of Microsoft's OpenAI partnership on cloud revenues",
            "AI Investment Tsunami": "How are AI investments reshaping Big Tech company valuations and strategies?",
            "Custom Query": ""
        }
    
    selected_query = st.selectbox("Choose analysis query:", list(market_queries.keys()))
    
    if selected_query == "Custom Query":
        query = st.text_area("Enter your multi-hop analysis question:", height=120, 
                           placeholder="Ask complex questions requiring multi-source analysis...")
    else:
        query = market_queries[selected_query]
        if query:
            st.text_area("Selected Query:", value=query, height=120, disabled=True)
    
    # Enhanced execution button
    col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
    
    with col_btn1:
        execute_analysis = st.button("🚀 Execute Multi-Hop Analysis", type="primary", use_container_width=True)
    
    with col_btn2:
        if st.button("⭐ Save Query"):
            if query and query != "":
                if "saved_queries" not in st.session_state:
                    st.session_state.saved_queries = []
                if query not in st.session_state.saved_queries:
                    st.session_state.saved_queries.append(query)
                    st.success("Query saved!")
    
    with col_btn3:
        if st.button("📋 Clear"):
            st.rerun()

with col2:
    st.subheader("📊 Quick Stock Lookup")
    
    symbol = st.text_input("Stock Symbol:", placeholder="AAPL, MSFT, GOOGL...")
    
    if st.button("📈 Get Real-Time Data", use_container_width=True):
        if symbol:
            try:
                response = requests.get(f"{API_URL}/market/{symbol.upper()}")
                if response.status_code == 200:
                    data = response.json()
                    if "error" not in data:
                        st.success(f"**{data['company_name']}** ({data['symbol']})")
                        
                        col_price1, col_price2 = st.columns(2)
                        with col_price1:
                            st.metric("Current Price", f"${data['current_price']}")
                        with col_price2:
                            st.metric("Change", f"{data['change']}", f"{data['change_percent']}%")
                        
                        st.write(f"📊 Volume: {data['volume']:,}")
                        st.write(f"🏢 Sector: {data.get('sector', 'N/A')}")
                    else:
                        st.error(data['error'])
            except:
                st.error("Failed to fetch stock data")
    
    st.divider()
    
    # Saved queries
    if "saved_queries" in st.session_state and st.session_state.saved_queries:
        st.subheader("⭐ Saved Queries")
        for i, saved_query in enumerate(st.session_state.saved_queries[-3:]):  # Show last 3
            if st.button(f"📌 {saved_query[:30]}...", key=f"saved_{i}"):
                st.session_state.selected_query = saved_query
                st.rerun()

# Execute analysis
if execute_analysis and query:
    with st.spinner("🧠 Processing multi-hop analysis..."):
        start_time = time.time()
        try:
            response = requests.post(
                f"{API_URL}/query",
                headers={"Content-Type": "application/json"},
                json={"query": query}
            )
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Success metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                with col_metric2:
                    st.metric("📈 Sources Used", len(data.get('sources', [])))
                with col_metric3:
                    st.metric("🔍 Evidence Points", len(data.get('retrieved_data', [])))
                
                st.success(f"✅ {category} Analysis Complete!")
                
                # Main analysis result
                st.subheader("💡 Multi-Hop Analysis Result")
                st.markdown(f"""
                <div class="category-card">
                    {data['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Reasoning chain
                if data.get("reasoning_steps"):
                    st.subheader("🔄 Multi-Hop Reasoning Chain")
                    for i, step in enumerate(data["reasoning_steps"], 1):
                        st.text(f"{i}. {step}")
                
                # Retrieved data with better formatting
                if data.get("retrieved_data"):
                    st.subheader("📊 Retrieved Evidence")
                    for i, item in enumerate(data["retrieved_data"], 1):
                        with st.expander(f"🔍 Evidence {i}: {item['sub_question']}", expanded=False):
                            st.info(f"**Data Source:** {item['data_source']}")
                            st.code(item['content'], language='json')
                
                # Sources summary
                if data.get("sources"):
                    st.subheader("📡 Data Sources")
                    source_cols = st.columns(min(len(data['sources']), 4))
                    for i, source in enumerate(data['sources']):
                        with source_cols[i % 4]:
                            st.success(f"📈 {source}")
                
                # Export functionality
                st.subheader("📥 Export Results")
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    st.download_button(
                        "📄 Download JSON",
                        json.dumps(data, indent=2),
                        f"rag_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                with col_export2:
                    report = f"""# {category} Analysis Report

**Query:** {query}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Processing Time:** {processing_time:.2f}s

## Analysis Result
{data['answer']}

## Sources
{', '.join(data.get('sources', []))}

## Reasoning Steps
{chr(10).join([f"{i}. {step}" for i, step in enumerate(data.get('reasoning_steps', []), 1)])}
"""
                    st.download_button(
                        "📊 Download Report",
                        report,
                        f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown"
                    )
                        
            else:
                st.error(f"❌ API Error: {response.status_code}")
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
elif execute_analysis:
    st.warning("⚠️ Please select or enter a query")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 <strong>Advanced Multi-Hop RAG Agent</strong> | Finance • Sports • Business Intelligence</p>
    <p>Real-World Data Analysis | Multi-Source Evidence Synthesis | NSKAI Bootcamp 2025</p>
</div>
""", unsafe_allow_html=True)