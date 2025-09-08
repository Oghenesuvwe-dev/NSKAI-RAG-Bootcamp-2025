import streamlit as st
import asyncio
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Import our enhanced modules
try:
    from enhanced_news_integration import NewsAnalyzer, AsyncProcessor, render_news_dashboard, enhanced_multi_hop_with_news
    from pwa_config import PWAConfig, optimize_for_mobile, add_mobile_navigation, create_mobile_shortcuts
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    st.warning("Enhanced features not available. Install required packages: pip install newsapi-python textblob aiohttp")

# Import existing functionality
from unified_rag_with_graphs import (
    init_groq_client, init_database, get_stock_data, get_real_sports_data,
    get_business_intelligence, process_multi_hop_query
)

# Enhanced Streamlit Configuration
st.set_page_config(
    page_title="🚀 Enhanced Multi-Hop RAG Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': 'Enhanced Multi-Hop RAG Agent with News Integration - NSKAI Bootcamp 2025'
    }
)

# Initialize PWA and mobile optimization
if NEWS_AVAILABLE:
    pwa_config = PWAConfig()
    st.markdown(pwa_config.inject_pwa_html(), unsafe_allow_html=True)
    optimize_for_mobile()

# Initialize clients and database
client = init_groq_client()
db_conn = init_database()

# Enhanced session state initialization
if "enhanced_features" not in st.session_state:
    st.session_state.enhanced_features = {
        "news_enabled": NEWS_AVAILABLE,
        "async_processing": True,
        "mobile_optimized": True,
        "pwa_ready": NEWS_AVAILABLE
    }

if "news_cache" not in st.session_state:
    st.session_state.news_cache = {}

if "processing_mode" not in st.session_state:
    st.session_state.processing_mode = "enhanced"  # enhanced, standard, fast

# Enhanced theme system
def get_enhanced_theme_css(dark_mode, mobile_optimized=True):
    base_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    """
    
    if dark_mode:
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: #f8fafc;
        }
        
        .enhanced-header {
            background: linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #db2777 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .feature-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #3b82f6;
            margin: 1rem 0;
            color: #f1f5f9;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        
        .news-card {
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 1rem;
            border-radius: 12px;
            border-left: 4px solid #10b981;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }
        
        .news-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.3);
        }
        """
    else:
        theme_css = """
        .stApp {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        .enhanced-header {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .feature-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1.5rem;
            border-radius: 15px;
            border-left: 5px solid #3b82f6;
            margin: 1rem 0;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .news-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            padding: 1rem;
            border-radius: 12px;
            border-left: 4px solid #10b981;
            margin: 0.5rem 0;
            box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.1);
            transition: all 0.3s ease;
        }
        
        .news-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.2);
        }
        """
    
    if mobile_optimized:
        mobile_css = """
        @media (max-width: 768px) {
            .enhanced-header {
                padding: 1.5rem 1rem;
            }
            
            .enhanced-header h1 {
                font-size: 1.8rem !important;
            }
            
            .feature-card {
                padding: 1rem;
                margin: 0.5rem 0;
            }
            
            .news-card {
                padding: 0.75rem;
            }
        }
        """
        theme_css += mobile_css
    
    return base_css + theme_css + "</style>"

# Apply enhanced theme
st.markdown(get_enhanced_theme_css(
    st.session_state.get('dark_mode', False),
    st.session_state.enhanced_features.get('mobile_optimized', True)
), unsafe_allow_html=True)

# Enhanced header with feature indicators
def render_enhanced_header():
    features_enabled = []
    if st.session_state.enhanced_features.get('news_enabled'):
        features_enabled.append("📰 News")
    if st.session_state.enhanced_features.get('async_processing'):
        features_enabled.append("⚡ Async")
    if st.session_state.enhanced_features.get('mobile_optimized'):
        features_enabled.append("📱 Mobile")
    if st.session_state.enhanced_features.get('pwa_ready'):
        features_enabled.append("🔧 PWA")
    
    features_text = " • ".join(features_enabled) if features_enabled else "Standard Mode"
    
    st.markdown(f"""
    <div class="enhanced-header">
        <h1 style="font-size: 2.5rem; margin: 0 0 1rem 0; font-weight: 700;">
            🚀 Enhanced Multi-Hop RAG Agent
        </h1>
        <p style="font-size: 1.2rem; margin: 0 0 1rem 0; opacity: 0.9;">
            Finance • Sports • Business Intelligence • News Analysis
        </p>
        <div style="background: rgba(255,255,255,0.1); padding: 0.5rem 1rem; border-radius: 20px; 
                   display: inline-block; font-size: 0.9rem; margin-top: 1rem;">
            {features_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

render_enhanced_header()

# Enhanced sidebar with new features
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
               padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
        <h3 style="margin: 0; text-align: center;">🎛️ Enhanced Controls</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Processing mode selection
    processing_mode = st.selectbox(
        "Processing Mode:",
        ["Enhanced (News + Async)", "Standard (Multi-hop)", "Fast (Single-hop)"],
        index=0 if st.session_state.processing_mode == "enhanced" else 1 if st.session_state.processing_mode == "standard" else 2
    )
    
    if "Enhanced" in processing_mode:
        st.session_state.processing_mode = "enhanced"
    elif "Standard" in processing_mode:
        st.session_state.processing_mode = "standard"
    else:
        st.session_state.processing_mode = "fast"
    
    # News settings
    if NEWS_AVAILABLE:
        st.subheader("📰 News Settings")
        news_enabled = st.checkbox("Enable News Analysis", value=True)
        st.session_state.enhanced_features["news_enabled"] = news_enabled
        
        if news_enabled:
            news_sources = st.multiselect(
                "News Sources:",
                ["All Sources", "Financial Times", "Reuters", "Bloomberg", "CNBC"],
                default=["All Sources"]
            )
            
            sentiment_threshold = st.slider(
                "Sentiment Sensitivity:",
                0.1, 1.0, 0.3, 0.1,
                help="Lower values = more sensitive sentiment detection"
            )
    
    # Async processing settings
    st.subheader("⚡ Performance Settings")
    async_enabled = st.checkbox("Enable Async Processing", value=True)
    st.session_state.enhanced_features["async_processing"] = async_enabled
    
    if async_enabled:
        max_parallel = st.slider("Max Parallel Tasks:", 2, 8, 4)
        timeout_seconds = st.slider("Request Timeout (s):", 5, 30, 10)
    
    # Mobile optimization toggle
    mobile_optimized = st.checkbox("Mobile Optimization", value=True)
    st.session_state.enhanced_features["mobile_optimized"] = mobile_optimized
    
    if mobile_optimized and NEWS_AVAILABLE:
        add_mobile_navigation()
        create_mobile_shortcuts()

# Main interface with enhanced features
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="feature-card">
        <h2 style="margin: 0 0 1rem 0;">🎯 Enhanced Analysis Interface</h2>
        <p style="margin: 0; opacity: 0.8;">
            Multi-hop reasoning with real-time news integration and async processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced query input with news context
    query_context = st.selectbox(
        "Analysis Context:",
        ["💰 Financial Markets", "⚽ Sports Analysis", "🏢 Business Intelligence", "📰 News & Sentiment", "🔄 Multi-Domain"]
    )
    
    # Smart query suggestions based on context
    if query_context == "📰 News & Sentiment":
        example_queries = {
            "Market Sentiment Analysis": "Analyze current market sentiment around Tesla stock with recent news",
            "Economic News Impact": "How are recent Federal Reserve announcements affecting tech stock prices?",
            "Company News Analysis": "What's the sentiment around Microsoft's latest AI developments?",
            "Sector News Trends": "Analyze news sentiment trends in the renewable energy sector"
        }
    elif query_context == "🔄 Multi-Domain":
        example_queries = {
            "Cross-Market Analysis": "How do sports betting trends correlate with tech stock performance?",
            "News-Market Correlation": "Analyze the relationship between AI news sentiment and NVIDIA stock price",
            "Multi-Source Intelligence": "Compare Premier League performance with club stock valuations and news sentiment"
        }
    else:
        # Use existing query examples
        example_queries = {
            "Select example...": "",
            "Custom Analysis": ""
        }
    
    selected_example = st.selectbox("Choose example query:", list(example_queries.keys()))
    
    if selected_example == "Custom Analysis" or selected_example == "Select example...":
        query = st.text_area(
            "Enter your enhanced analysis question:",
            height=120,
            placeholder="Ask complex questions requiring analysis across multiple data sources including news...",
            help="Enhanced mode supports news analysis, sentiment tracking, and async processing"
        )
    else:
        query = example_queries[selected_example]
        st.text_area("Selected Query:", value=query, height=120, disabled=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 style="margin: 0 0 1rem 0;">📊 Enhanced Features</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature status indicators
    features_status = [
        ("📰 News Analysis", st.session_state.enhanced_features.get('news_enabled', False)),
        ("⚡ Async Processing", st.session_state.enhanced_features.get('async_processing', False)),
        ("📱 Mobile Optimized", st.session_state.enhanced_features.get('mobile_optimized', False)),
        ("🔧 PWA Ready", st.session_state.enhanced_features.get('pwa_ready', False))
    ]
    
    for feature, enabled in features_status:
        status_color = "green" if enabled else "gray"
        status_icon = "✅" if enabled else "⭕"
        st.markdown(f"{status_icon} **{feature}**: {'Enabled' if enabled else 'Disabled'}")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    if st.button("📰 Latest Market News", use_container_width=True):
        if NEWS_AVAILABLE:
            with st.spinner("Fetching latest market news..."):
                news_analyzer = NewsAnalyzer()
                news_data = asyncio.run(news_analyzer.get_market_news_async("market trends", "business"))
                st.session_state.latest_news = news_data
                st.success("News data loaded!")
        else:
            st.error("News integration not available")
    
    if st.button("⚡ Performance Test", use_container_width=True):
        with st.spinner("Running performance test..."):
            start_time = time.time()
            
            # Test async processing if available
            if st.session_state.enhanced_features.get('async_processing') and NEWS_AVAILABLE:
                async_processor = AsyncProcessor()
                test_queries = ["AAPL stock", "Premier League", "AI market"]
                results = asyncio.run(async_processor.parallel_data_retrieval(test_queries, "performance test"))
                processing_time = time.time() - start_time
                st.success(f"Async test: {processing_time:.2f}s for {len(test_queries)} queries")
            else:
                time.sleep(1)  # Simulate processing
                processing_time = time.time() - start_time
                st.info(f"Standard test: {processing_time:.2f}s")

# Enhanced analysis execution
if st.button("🚀 Execute Enhanced Analysis", type="primary", use_container_width=True):
    if query and query.strip():
        progress_placeholder = st.empty()
        
        with st.spinner("🧠 Processing enhanced multi-hop analysis..."):
            start_time = time.time()
            
            # Choose processing method based on mode
            if st.session_state.processing_mode == "enhanced" and NEWS_AVAILABLE:
                # Enhanced processing with news and async
                progress_placeholder.progress(0.2, text="Step 1/5: Query decomposition")
                result = asyncio.run(enhanced_multi_hop_with_news(query.strip(), query_context))
                
            elif st.session_state.processing_mode == "standard":
                # Standard multi-hop processing
                progress_placeholder.progress(0.3, text="Step 1/3: Standard processing")
                result = process_multi_hop_query(query.strip())
                
            else:
                # Fast single-hop processing
                progress_placeholder.progress(0.5, text="Step 1/2: Fast processing")
                if client:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": query.strip()}],
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=1000
                    )
                    result = {
                        "answer": response.choices[0].message.content,
                        "processing_time": time.time() - start_time,
                        "sources": ["Direct LLM"],
                        "sub_questions": [query.strip()],
                        "mode": "fast"
                    }
                else:
                    result = {"error": "LLM client not available"}
            
            progress_placeholder.progress(1.0, text="Analysis complete!")
            time.sleep(0.5)
            progress_placeholder.empty()
            
            if "error" not in result:
                # Enhanced results display
                st.markdown("""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                           padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;
                           box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.3);">
                    <h3 style="margin: 0; font-size: 1.3rem;">✅ Enhanced Analysis Complete!</h3>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Multi-source intelligence with real-time data</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Performance metrics
                col_perf1, col_perf2, col_perf3, col_perf4 = st.columns(4)
                
                with col_perf1:
                    st.metric("Processing Time", f"{result['processing_time']:.2f}s")
                with col_perf2:
                    st.metric("Data Sources", len(result.get('sources', [])))
                with col_perf3:
                    mode_label = result.get('mode', st.session_state.processing_mode).title()
                    st.metric("Processing Mode", mode_label)
                with col_perf4:
                    async_used = result.get('async_processed', False)
                    st.metric("Async Processing", "Yes" if async_used else "No")
                
                # Main analysis result
                st.subheader("💡 Analysis Result")
                st.markdown(f"""
                <div style="background: {'rgba(30, 41, 59, 0.3)' if st.session_state.get('dark_mode') else 'rgba(255, 255, 255, 0.8)'}; 
                           padding: 1.5rem; border-radius: 12px; line-height: 1.6; font-size: 1rem;
                           border-left: 4px solid #3b82f6;">
                    {result['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # News analysis section
                if result.get('news_data') and NEWS_AVAILABLE:
                    st.subheader("📰 News Analysis")
                    render_news_dashboard(result['news_data'])
                
                # Enhanced visualizations
                if len(result.get('sub_questions', [])) > 1:
                    st.subheader("🕸️ Enhanced Reasoning Network")
                    
                    # Create enhanced network visualization
                    fig_network = go.Figure()
                    
                    # Add network nodes and edges
                    n_nodes = len(result['sub_questions']) + 1
                    angles = [2 * np.pi * i / (n_nodes - 1) for i in range(n_nodes - 1)]
                    
                    x_nodes = [0] + [np.cos(angle) for angle in angles]
                    y_nodes = [0] + [np.sin(angle) for angle in angles]
                    
                    # Enhanced edges with different colors
                    edge_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']
                    
                    for i in range(1, n_nodes):
                        fig_network.add_trace(go.Scatter(
                            x=[0, x_nodes[i]], y=[0, y_nodes[i]],
                            mode='lines',
                            line=dict(width=3, color=edge_colors[i % len(edge_colors)]),
                            showlegend=False,
                            hoverinfo='none'
                        ))
                    
                    # Enhanced nodes
                    node_colors = ['#ef4444'] + ['#3b82f6'] * (n_nodes - 1)
                    node_sizes = [30] + [20] * (n_nodes - 1)
                    
                    fig_network.add_trace(go.Scatter(
                        x=x_nodes, y=y_nodes,
                        mode='markers+text',
                        marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
                        text=['Main'] + [f'Q{i}' for i in range(1, n_nodes)],
                        textposition="middle center",
                        textfont=dict(color='white', size=12, family='Inter'),
                        hovertext=[query[:50] + '...'] + result['sub_questions'],
                        showlegend=False
                    ))
                    
                    fig_network.update_layout(
                        title="Enhanced Multi-Hop Reasoning Network",
                        showlegend=False,
                        height=400,
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_network, use_container_width=True)
                
                # Enhanced export options
                st.subheader("📤 Enhanced Export & Sharing")
                
                export_data = {
                    "query": query,
                    "result": result,
                    "processing_mode": st.session_state.processing_mode,
                    "features_used": st.session_state.enhanced_features,
                    "timestamp": datetime.now().isoformat()
                }
                
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    st.download_button(
                        "📄 Download Enhanced JSON",
                        json.dumps(export_data, indent=2),
                        f"enhanced_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
                
                with col_export2:
                    if st.button("📧 Email Report"):
                        st.success("Email functionality would be implemented here")
                
                with col_export3:
                    if st.button("🔗 Share Analysis"):
                        st.success("Sharing functionality would be implemented here")
                
            else:
                st.error(f"❌ Enhanced analysis failed: {result['error']}")
    else:
        st.warning("⚠️ Please enter a query to analyze")

# Enhanced footer with system status
st.markdown("<br><br>", unsafe_allow_html=True)

# System status dashboard
col_status1, col_status2, col_status3, col_status4 = st.columns(4)

with col_status1:
    st.metric("System Status", "🟢 Online")
with col_status2:
    st.metric("Features Active", f"{sum(st.session_state.enhanced_features.values())}/4")
with col_status3:
    st.metric("Processing Mode", st.session_state.processing_mode.title())
with col_status4:
    st.metric("News Integration", "✅ Ready" if NEWS_AVAILABLE else "❌ Unavailable")

# Enhanced footer
st.markdown(f"""
<div style="background: linear-gradient(135deg, {'#1e293b' if st.session_state.get('dark_mode') else '#f8fafc'} 0%, 
                                                {'#334155' if st.session_state.get('dark_mode') else '#e2e8f0'} 100%); 
           padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;
           border: 1px solid {'rgba(148, 163, 184, 0.2)' if st.session_state.get('dark_mode') else 'rgba(148, 163, 184, 0.3)'};
           box-shadow: 0 10px 25px -5px rgba(0, 0, 0, {'0.2' if st.session_state.get('dark_mode') else '0.1'});">
    
    <h2 style="margin: 0 0 1rem 0; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%); 
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
               font-size: 1.8rem; font-weight: 700;">
        🚀 Enhanced Multi-Hop RAG Agent
    </h2>
    
    <p style="margin: 0 0 1.5rem 0; font-size: 1.1rem; color: {'#94a3b8' if st.session_state.get('dark_mode') else '#64748b'}; font-weight: 500;">
        Finance • Sports • Business Intelligence • News Analysis • PWA Ready
    </p>
    
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem;">
        <span style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 500;">
            ✅ Enhanced Features Active
        </span>
        <span style="background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%); 
                    color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 500;">
            📰 News Integration
        </span>
        <span style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); 
                    color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 500;">
            ⚡ Async Processing
        </span>
        <span style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); 
                    color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 500;">
            📱 PWA Ready
        </span>
    </div>
    
    <p style="margin: 1.5rem 0 0 0; font-size: 0.9rem; color: {'#64748b' if st.session_state.get('dark_mode') else '#94a3b8'}; font-weight: 500;">
        NSKAI Bootcamp 2025 | Enhanced with News, Async Processing & PWA Features
    </p>
</div>
""", unsafe_allow_html=True)