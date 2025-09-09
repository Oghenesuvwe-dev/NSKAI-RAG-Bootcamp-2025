import streamlit as st
import os
from core.multi_hop_engine import MultiHopRAG

st.set_page_config(page_title="Multi-Hop RAG", page_icon="🎯", layout="centered")

st.title("🎯 Multi-Hop RAG Agent")
st.write("Ask complex questions about Sports, Business, and Finance")

@st.cache_resource
def init_rag():
    return MultiHopRAG()

# Check API key
if not os.getenv("GROQ_API_KEY"):
    st.error("Please set GROQ_API_KEY environment variable")
    st.stop()

rag = init_rag()

examples = [
    "How did Liverpool's recent performance affect their Champions League chances vs Arsenal?",
    "Compare Microsoft's AI investment impact on stock vs Google's strategy",
    "What's the correlation between tech stocks and sports team valuations?"
]

selected = st.selectbox("Choose example or write custom:", ["Custom..."] + examples)
query = st.text_area("Your question:", value="" if selected == "Custom..." else selected, height=100)

if st.button("🔍 Analyze", type="primary"):
    if query.strip():
        with st.spinner("Processing multi-hop analysis..."):
            result = rag.process_query(query)
        
        st.success("✅ Analysis Complete")
        st.subheader("Answer")
        st.write(result["answer"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']:.0%}")
        with col2:
            st.metric("Sources", len(result["sources"]))
        
        with st.expander("🧠 Reasoning Process"):
            st.write("**Sub-questions analyzed:**")
            for i, sq in enumerate(result["sub_questions"], 1):
                st.write(f"{i}. {sq}")
            
            st.write("**Evidence gathered:**")
            for evidence in result["evidence"]:
                st.write(f"• **{evidence['source'].title()}**: {str(evidence['data'])[:200]}...")
            
            st.write(f"**Sources**: {', '.join(result['sources'])}")
    else:
        st.warning("Please enter a question")