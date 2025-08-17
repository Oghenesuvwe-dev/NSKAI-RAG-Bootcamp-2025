from langchain_community.vectorstores import Chroma
import os

class SimpleEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
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
    
    embeddings = SimpleEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore