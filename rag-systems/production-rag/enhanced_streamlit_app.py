import streamlit as st
import requests
import json
import time
from datetime import datetime
import pandas as pd

st.set_page_config(
    page_title="Advanced RAG Research Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8010"

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "bookmarks" not in st.session_state:
    st.session_state.bookmarks = []

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def query_api(query_text):
    try:
        start_time = time.time()
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query_text}, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result["client_time"] = end_time - start_time
            return True, result
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def get_metrics():
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.query-card {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #007bff;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 Advanced RAG Multi-Hop Research Agent")
st.markdown("**Production-Ready with Caching, Rate Limiting & Enhanced UI**")

# Sidebar
with st.sidebar:
    st.header("🛠️ System Dashboard")
    
    # API Status
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("✅ Production API Online")
        st.markdown(f"""
        <div class="metric-card">
            <h4>System Status</h4>
            <p>Version: {health_data.get('version', 'N/A')}</p>
            <p>Documents: {health_data.get('documents', 0)}</p>
            <p>Cache: {'✅' if health_data.get('cache_enabled') else '❌'}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ API Offline")
        st.code("python -m uvicorn production_rag:app --port 8009")
    
    st.divider()
    
    # Metrics
    if is_healthy:
        st.header("📊 Performance Metrics")
        metrics = get_metrics()
        if metrics:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", metrics.get("documents_indexed", 0))
            with col2:
                cache_info = metrics.get("cache", {})
                st.metric("Cache Keys", cache_info.get("keys", 0))
    
    st.divider()
    
    # Query History
    st.header("📝 Query History")
    if st.session_state.query_history:
        for i, (timestamp, query, success) in enumerate(reversed(st.session_state.query_history[-5:])):
            status = "✅" if success else "❌"
            if st.button(f"{status} {query[:30]}...", key=f"hist_{i}"):
                st.session_state.selected_query = query
    else:
        st.info("No queries yet")
    
    if st.button("🗑️ Clear History"):
        st.session_state.query_history = []
        st.rerun()

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Multi-Hop Query Interface")
    
    # Example queries with categories
    st.subheader("📚 Example Queries by Category")
    
    categories = {
        "🏈 Sports Analysis": [
            "How did Arsenal's victory over City affect Liverpool's title chances?",
            "What impact did Messi's transfer have on PSG's Champions League performance?"
        ],
        "💼 Business Intelligence": [
            "How did Company X's Q3 earnings affect supplier relationships?",
            "What was the impact of OpenAI investment on Microsoft Azure revenue?"
        ],
        "🌍 Economic Analysis": [
            "How did the Noto Peninsula earthquake affect Japan's semiconductor production?",
            "What was the impact of Brexit on UK-EU trade relationships?"
        ]
    }
    
    selected_category = st.selectbox("Choose category:", list(categories.keys()))
    selected_example = st.selectbox("Choose example:", [""] + categories[selected_category])
    
    # Query input
    query_input = st.text_area(
        "Enter your multi-hop research query:",
        value=selected_example if selected_example else st.session_state.get("selected_query", ""),
        height=120,
        placeholder="Ask a complex question requiring multi-source analysis..."
    )
    
    # Query controls
    col_a, col_b, col_c = st.columns([2, 1, 1])
    
    with col_a:
        run_query = st.button("🔍 Run Analysis", type="primary", disabled=not is_healthy)
    
    with col_b:
        if st.button("⭐ Bookmark") and query_input.strip():
            if query_input not in st.session_state.bookmarks:
                st.session_state.bookmarks.append(query_input)
                st.success("Bookmarked!")
    
    with col_c:
        if st.button("📋 Export") and hasattr(st.session_state, 'last_result'):
            result_json = json.dumps(st.session_state.last_result, indent=2)
            st.download_button("Download JSON", result_json, "rag_result.json", "application/json")

with col2:
    st.header("🔧 System Features")
    st.markdown("""
    **Production Features:**
    - 🚀 Rate limiting (10 queries/min)
    - 💾 Redis caching (1hr TTL)
    - 📊 Performance monitoring
    - 🔍 Multi-format document support
    - 🧠 ChromaDB vector search
    - 📝 Query history & bookmarks
    """)
    
    # Bookmarks
    if st.session_state.bookmarks:
        st.subheader("⭐ Bookmarked Queries")
        for i, bookmark in enumerate(st.session_state.bookmarks):
            if st.button(f"📌 {bookmark[:40]}...", key=f"book_{i}"):
                st.session_state.selected_query = bookmark
                st.rerun()

# Process query
if run_query and query_input.strip():
    with st.spinner("🧠 Processing multi-hop analysis..."):
        success, result = query_api(query_input.strip())
        
        # Add to history
        st.session_state.query_history.append((
            datetime.now().strftime("%H:%M:%S"),
            query_input.strip(),
            success
        ))
        
        if success:
            st.session_state.last_result = result
            st.success(f"✅ Analysis completed in {result.get('processing_time', 0):.2f}s")
        else:
            st.error(f"❌ Analysis failed: {result}")

# Results display
if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
    st.divider()
    st.header("📋 Analysis Results")
    
    result = st.session_state.last_result
    
    # Performance metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Processing Time", f"{result.get('processing_time', 0):.2f}s")
    with col2:
        st.metric("Sources Used", len(result.get('sources', [])))
    with col3:
        st.metric("Evidence Chunks", len(result.get('retrieved_documents', [])))
    
    # Main answer
    st.subheader("💡 Synthesized Answer")
    st.markdown(f"""
    <div class="query-card">
        {result['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Reasoning chain
    st.subheader("🔄 Multi-Hop Reasoning Chain")
    for i, step in enumerate(result['reasoning_steps'], 1):
        st.text(f"{i}. {step}")
    
    # Sources and evidence
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Sources")
        for source in result['sources']:
            st.text(f"📄 {source}")
    
    with col2:
        st.subheader("🔍 Evidence Summary")
        for doc in result['retrieved_documents']:
            st.text(f"• {doc['source']}: {doc['similarity_score']:.3f}")
    
    # Detailed evidence
    with st.expander("📖 Detailed Evidence"):
        for i, doc in enumerate(result['retrieved_documents'], 1):
            st.markdown(f"**Evidence {i}: {doc['source']}**")
            st.text(f"Similarity: {doc.get('similarity_score', 0):.3f}")
            st.text(doc['content'][:300] + "..." if len(doc['content']) > 300 else doc['content'])
            st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🚀 Advanced RAG Multi-Hop Research Agent v9.0 | Production Ready</p>
    <p>Features: Rate Limiting • Caching • Multi-Format Processing • Vector Search</p>
</div>
""", unsafe_allow_html=True)