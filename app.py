import streamlit as st
import os
from dotenv import load_dotenv
from helpers.youtubeloader import load_youtube_transcript
from helpers.chunker import chunk_documents
from helpers.vectorstore import create_vectorstore
from helpers.retriever import create_retriever
from helpers.chain import create_rag_chain

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="YouTube Q&A Chatbot",
    page_icon="💬",
    layout="wide"
)

st.title("YouTube Video Q&A with RAG 💬")
st.markdown("Ask questions about any YouTube video using AI-powered retrieval!")

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for video processing
with st.sidebar:
    st.header("Video Processing")
    youtube_url = st.text_input("Enter YouTube URL:")
    
    if st.button("Process Video"):
        if youtube_url:
            with st.spinner("Loading transcript..."):
                try:
                    # Load transcript
                    documents = load_youtube_transcript(youtube_url)
                    
                    # Chunk documents
                    chunks = chunk_documents(documents)
                    
                    # Create vectorstore
                    vectorstore = create_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    
                    # Create retriever
                    retriever = create_retriever(vectorstore)
                    st.session_state.retriever = retriever
                    
                    # Create RAG chain
                    chain = create_rag_chain(retriever)
                    st.session_state.chain = chain
                    
                    st.success("Video processed successfully!")
                    st.session_state.messages = []  # Clear previous messages
                    
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
        else:
            st.error("Please enter a YouTube URL")

# Main chat interface
if st.session_state.chain:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chain.invoke({"question": prompt})
                    answer = response.get("answer", "I couldn't generate an answer.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
else:
    st.info("👈 Please process a YouTube video first using the sidebar.")