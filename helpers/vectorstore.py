import numpy as np
from scipy.spatial.distance import cosine
import os

class SimpleVectorStore:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = np.array(embeddings)
    
    def similarity_search(self, query_embedding, k=5):
        # Calculate cosine similarity
        similarities = []
        query_embedding = np.array(query_embedding)
        
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = 1 - cosine(query_embedding, doc_embedding)
            similarities.append((similarity, i))
        
        # Sort by similarity and return top k
        similarities.sort(reverse=True)
        return [self.documents[i] for _, i in similarities[:k]]
    
    def as_retriever(self, search_kwargs=None):
        k = search_kwargs.get('k', 5) if search_kwargs else 5
        return SimpleRetriever(self, k)

class SimpleRetriever:
    def __init__(self, vectorstore, k):
        self.vectorstore = vectorstore
        self.k = k
    
    def get_relevant_documents(self, query):
        # Simple hash-based embedding for query
        query_embedding = [hash(query[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(query), 384), 10)]
        while len(query_embedding) < 384:
            query_embedding.append(0.0)
        
        return self.vectorstore.similarity_search(query_embedding, self.k)

def create_vectorstore(chunks):
    """Create simple in-memory vector store"""
    
    # Create simple hash-based embeddings
    embeddings = []
    for chunk in chunks:
        text = chunk.page_content
        embedding = [hash(text[i:i+10]) % 1000 / 1000.0 for i in range(0, min(len(text), 384), 10)]
        while len(embedding) < 384:
            embedding.append(0.0)
        embeddings.append(embedding[:384])
    
    # Create simple vector store
    vectorstore = SimpleVectorStore(chunks, embeddings)
    return vectorstore