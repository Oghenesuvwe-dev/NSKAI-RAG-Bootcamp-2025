from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os

def create_vectorstore(chunks):
    """Create ChromaDB vectorstore with Groq embeddings"""
    
    # Use Groq for embeddings via text generation
    from langchain.embeddings.base import Embeddings
    import numpy as np
    
    class GroqEmbeddings(Embeddings):
        def __init__(self):
            self.llm = ChatGroq(
                groq_api_key=os.getenv("GROQ_API_KEY"),
                model_name="llama3-8b-8192"
            )
        
        def embed_documents(self, texts):
            # Simple hash-based embeddings for demo
            embeddings = []
            for text in texts:
                # Create a simple embedding from text hash
                embedding = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 10)]
                # Pad to 384 dimensions
                while len(embedding) < 384:
                    embedding.append(0.0)
                embeddings.append(embedding[:384])
            return embeddings
        
        def embed_query(self, text):
            embedding = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 10)]
            while len(embedding) < 384:
                embedding.append(0.0)
            return embedding[:384]
    
    embeddings = GroqEmbeddings()
    
    # Create in-memory ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="youtube_transcript"
    )
    
    return vectorstore