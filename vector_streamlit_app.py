import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="Vector RAG Multi-Hop Agent",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_BASE_URL = "http://localhost:8007"

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def query_api(query_text):
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"query": query_text},
            timeout=30
        )
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

def get_documents():
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

# Main UI
st.title("🔍 Vector RAG Multi-Hop Research Agent")
st.markdown("**Enhanced with ChromaDB Vector Search & Semantic Embeddings**")

# Sidebar
with st.sidebar:
    st.header("🛠️ System Status")
    
    # API Health Check
    is_healthy, health_data = check_api_health()
    if is_healthy:
        st.success("✅ API Connected")
        st.json(health_data)
    else:
        st.error("❌ API Disconnected")
        st.warning("Start the API: `python -m uvicorn vector_rag_engine:app --port 8007`")
    
    st.divider()
    
    # Document Status
    st.header("📚 Document Database")
    if is_healthy:
        docs_info = get_documents()
        if docs_info:
            st.metric("Documents", docs_info.get("total_documents", 0))
            st.metric("Vector Chunks", docs_info.get("total_chunks", 0))
            
            if "documents" in docs_info:
                st.subheader("Indexed Files:")
                for doc in docs_info["documents"]:
                    st.text(f"📄 {doc['filename']} ({doc['chunks']} chunks)")
        else:
            st.warning("No documents indexed")
    
    st.divider()
    
    # Reindex Button
    if st.button("🔄 Reindex Documents"):
        with st.spinner("Reindexing..."):
            try:
                response = requests.post(f"{API_BASE_URL}/reindex", timeout=30)
                if response.status_code == 200:
                    st.success("Documents reindexed!")
                    st.rerun()
                else:
                    st.error("Reindex failed")
            except:
                st.error("Connection error")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Multi-Hop Query Interface")
    
    # Example queries
    st.subheader("📝 Example Queries")
    examples = [
        "How did Arsenal's victory over City affect Liverpool's title chances?",
        "What was the impact of Company X's Q3 earnings on supplier relationships?",
        "How did the Noto Peninsula earthquake affect Japan's semiconductor production?",
        "What are the connections between OpenAI investment and Microsoft Azure revenue?"
    ]
    
    selected_example = st.selectbox("Choose an example:", [""] + examples)
    
    # Query input
    query_input = st.text_area(
        "Enter your multi-hop research query:",
        value=selected_example,
        height=100,
        placeholder="Ask a complex question that requires information from multiple sources..."
    )
    
    # Query button
    if st.button("🔍 Run Multi-Hop Analysis", type="primary", disabled=not is_healthy):
        if query_input.strip():
            with st.spinner("🧠 Processing multi-hop query..."):
                success, result = query_api(query_input.strip())
                
                if success:
                    st.session_state.last_result = result
                else:
                    st.error(f"Query failed: {result}")
        else:
            st.warning("Please enter a query")

with col2:
    st.header("🔧 Vector Search Info")
    st.info("""
    **Enhanced Features:**
    - 🎯 ChromaDB vector database
    - 🧠 Semantic embeddings (all-MiniLM-L6-v2)
    - 📄 Document chunking with overlap
    - 🔍 Similarity-based retrieval
    - 🔗 Multi-hop reasoning chains
    """)
    
    st.header("📊 How It Works")
    st.markdown("""
    1. **Query Decomposition** - Break complex questions into sub-queries
    2. **Vector Search** - Find semantically similar document chunks
    3. **Multi-Hop Retrieval** - Gather evidence from multiple sources
    4. **Synthesis** - Combine evidence into coherent answer
    """)

# Results display
if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
    st.divider()
    st.header("📋 Analysis Results")
    
    result = st.session_state.last_result
    
    # Main answer
    st.subheader("💡 Synthesized Answer")
    st.write(result["answer"])
    
    # Reasoning steps
    st.subheader("🔄 Multi-Hop Reasoning Steps")
    for step in result["reasoning_steps"]:
        st.text(f"• {step}")
    
    # Sources
    if result["sources"]:
        st.subheader("📚 Sources Used")
        for source in result["sources"]:
            st.text(f"📄 {source}")
    
    # Retrieved documents
    with st.expander("🔍 Retrieved Document Chunks"):
        for i, doc in enumerate(result["retrieved_documents"], 1):
            st.markdown(f"**Chunk {i}: {doc['source']} ({doc.get('chunk_info', 'N/A')})**")
            st.text(f"Similarity Score: {doc.get('similarity_score', 0):.3f}")
            st.text(doc["content"])
            st.divider()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Vector RAG Multi-Hop Research Agent v7.0 | Enhanced with ChromaDB & Semantic Search</p>
</div>
""", unsafe_allow_html=True)