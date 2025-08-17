from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
import streamlit as st

def create_rag_chain(retriever):
    """Create the RAG chain with Groq LLM"""
    
    # Initialize Groq LLM
    api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",
        temperature=0.1
    )
    
    # Create prompt template
    template = """You are a helpful AI assistant that answers questions based on YouTube video transcripts.
    
    Context from the video:
    {context}
    
    Question: {question}
    
    Instructions:
    - Answer the question based only on the provided context from the video
    - If the context doesn't contain enough information to answer the question, say so
    - Be concise but comprehensive in your response
    - Use specific details from the video when possible
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Wrap in a function that returns dict for compatibility
    def rag_chain_wrapper(inputs):
        question = inputs.get("question", "")
        answer = chain.invoke(question)
        return {"answer": answer}
    
    return rag_chain_wrapper

def _format_docs(docs):
    """Format retrieved documents for the prompt"""
    return "\n\n".join([doc.page_content for doc in docs])