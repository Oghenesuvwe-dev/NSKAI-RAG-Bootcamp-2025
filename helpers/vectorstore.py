from langchain_community.vectorstores import Chroma
import os

class SimpleEmbeddings:
    def embed_documents(self, texts):
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            if not text or not text.strip():
                # Handle empty text with default embedding
                embedding = [0.0] * 384
            else:
                embedding = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 10)]
                while len(embedding) < 384:
                    embedding.append(0.0)
            embeddings.append(embedding[:384])
        return embeddings
    
    def embed_query(self, text):
        embedding = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 10)]
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]

def create_vectorstore(chunks):
    """Create ChromaDB vector store"""
    
    if not chunks:
        raise ValueError("No chunks provided to create vectorstore")
    
    # Filter out empty chunks
    valid_chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    
    if not valid_chunks:
        raise ValueError("No valid chunks with content found")
    
    embeddings = SimpleEmbeddings()
    vectorstore = Chroma.from_documents(valid_chunks, embeddings)
    return vectorstore