import streamlit as st
import requests
import json
import time
from datetime import datetime

st.set_page_config(
    page_title="Advanced RAG Research Agent",
    page_icon="🚀",
    layout="wide"
)

API_BASE_URL = "http://localhost:8006"  # Real RAG engine

def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def query_api(query_text):
    try:
        response = requests.post(f"{API_BASE_URL}/query", json={"query": query_text}, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API Error: {response.status_code}"
    except Exception as e:
        return False, f"Connection Error: {str(e)}"

# Header
st.title("🚀 Advanced RAG Multi-Hop Research Agent")
st.markdown("**Production System - Real Document Processing**")

# API Status
is_healthy, health_data = check_api_health()

col1, col2 = st.columns([3, 1])

with col1:
    if is_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")
        st.code("python -m uvicorn real_rag_engine:app --port 8006")

with col2:
    if is_healthy:
        st.metric("Documents", health_data.get("documents_loaded", 0))

# Example queries
st.subheader("📝 Multi-Hop Example Queries")
examples = [
    "How did Arsenal's victory over City affect Liverpool's title chances?",
    "What was the impact of Company X's Q3 earnings on supplier relationships?",
    "How did the Noto Peninsula earthquake affect Japan's semiconductor production?"
]

selected_example = st.selectbox("Choose an example:", [""] + examples)

# Query input
query_input = st.text_area(
    "Enter your multi-hop research query:",
    value=selected_example,
    height=100,
    placeholder="Ask a complex question requiring multi-source analysis..."
)

# Query button
if st.button("🔍 Run Multi-Hop Analysis", type="primary", disabled=not is_healthy):
    if query_input.strip():
        with st.spinner("🧠 Processing multi-hop query..."):
            success, result = query_api(query_input.strip())
            
            if success:
                st.success("✅ Analysis completed")
                
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
                with st.expander("🔍 Retrieved Evidence"):
                    for i, doc in enumerate(result["retrieved_documents"], 1):
                        st.markdown(f"**Evidence {i}: {doc['source']}**")
                        st.text(doc["content"])
                        st.divider()
            else:
                st.error(f"❌ Analysis failed: {result}")
    else:
        st.warning("Please enter a query")

# Footer
st.divider()
st.markdown("**Advanced RAG Multi-Hop Research Agent** | Real Document Processing | Multi-Source Evidence Synthesis")