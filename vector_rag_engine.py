from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from groq import Groq
import json
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import hashlib

load_dotenv()

app = FastAPI(title="Vector RAG Multi-Hop Research Agent", version="7.0.0")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB and embedding model
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    reasoning_steps: list
    sources: list
    retrieved_documents: list

def chunk_document(content: str, chunk_size: int = 500, overlap: int = 50):
    """Split document into overlapping chunks"""
    chunks = []
    start = 0
    while start < len(content):
        end = start + chunk_size
        chunk = content[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def initialize_vector_db():
    """Initialize ChromaDB collection with documents"""
    try:
        collection = chroma_client.get_collection("documents")
        return collection
    except:
        collection = chroma_client.create_collection("documents")
        
        # Load and process documents
        data_dir = Path("data")
        if not data_dir.exists():
            return collection
            
        documents = []
        metadatas = []
        ids = []
        
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                try:
                    content = file_path.read_text(encoding='utf-8')
                    chunks = chunk_document(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc_id = hashlib.md5(f"{file_path.name}_{i}".encode()).hexdigest()
                        documents.append(chunk)
                        metadatas.append({
                            "source": file_path.name,
                            "type": file_path.suffix[1:] if file_path.suffix else "txt",
                            "chunk_id": i,
                            "total_chunks": len(chunks)
                        })
                        ids.append(doc_id)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        return collection

# Initialize vector database
VECTOR_COLLECTION = initialize_vector_db()

def vector_search(query: str, top_k: int = 3):
    """Perform semantic vector search using ChromaDB"""
    try:
        results = VECTOR_COLLECTION.query(
            query_texts=[query],
            n_results=top_k
        )
        
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                search_results.append({
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                    "source": results['metadatas'][0][i]['source']
                })
        
        return search_results
    except Exception as e:
        print(f"Vector search error: {e}")
        return []

def multi_hop_retrieval(sub_query: str, main_query: str):
    """Perform vector-based retrieval for a sub-query"""
    search_query = f"{sub_query} {main_query}"
    results = vector_search(search_query, top_k=1)
    
    if results:
        best_match = results[0]
        return {
            "sub_question": sub_query,
            "document": best_match["source"],
            "similarity_score": 1 - best_match["distance"],
            "content": best_match["content"],
            "source": best_match["source"],
            "chunk_info": f"Chunk {best_match['metadata']['chunk_id'] + 1}/{best_match['metadata']['total_chunks']}"
        }
    else:
        return {
            "sub_question": sub_query,
            "document": "No relevant document found",
            "similarity_score": 0,
            "content": "No matching content found for this query.",
            "source": "None",
            "chunk_info": "N/A"
        }

@app.post("/query", response_model=QueryResponse)
def run_query(request: QueryRequest):
    try:
        # Check if vector database has documents
        collection_count = VECTOR_COLLECTION.count()
        if collection_count == 0:
            raise HTTPException(status_code=500, detail="No documents in vector database. Please add documents to the data/ directory and restart.")
        
        # Step 1: Query decomposition
        decomposition_prompt = f"""
        Decompose this complex query into 2-4 specific sub-questions for multi-hop reasoning:
        Query: "{request.query}"
        
        Each sub-question should target a specific piece of information needed to answer the main query.
        Return ONLY a JSON array of sub-questions, like:
        ["sub-question 1", "sub-question 2", "sub-question 3"]
        """
        
        decomp_response = client.chat.completions.create(
            messages=[{"role": "user", "content": decomposition_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [request.query]  # Fallback
        
        # Step 2: Multi-hop vector retrieval
        retrieved_docs = []
        reasoning_steps = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            doc_result = multi_hop_retrieval(sub_q, request.query)
            retrieved_docs.append(doc_result)
            reasoning_steps.append(f"Step {i}: {sub_q} → Retrieved from {doc_result['source']} (similarity: {doc_result['similarity_score']:.3f}) [{doc_result['chunk_info']}]")
        
        # Step 3: Evidence synthesis
        synthesis_prompt = f"""
        You are an expert analyst. Synthesize information from multiple document chunks to answer this query:
        
        Original Query: {request.query}
        
        Retrieved Evidence:
        {json.dumps([{k: v for k, v in doc.items() if k != 'content'} for doc in retrieved_docs], indent=2)}
        
        Document Contents:
        {chr(10).join([f"Source: {doc['source']} ({doc['chunk_info']}){chr(10)}{doc['content']}{chr(10)}" for doc in retrieved_docs])}
        
        Provide a comprehensive answer that:
        1. Directly addresses the original query
        2. Uses specific facts and numbers from the document chunks
        3. Shows clear reasoning connecting the evidence
        4. Cites sources appropriately with chunk information
        """
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        sources = list(set([doc["source"] for doc in retrieved_docs if doc["source"] != "None"]))
        
        return QueryResponse(
            answer=final_response.choices[0].message.content,
            reasoning_steps=reasoning_steps,
            sources=sources,
            retrieved_documents=retrieved_docs
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {
        "message": "Vector RAG Multi-Hop Research Agent", 
        "version": "7.0.0",
        "features": ["ChromaDB vector search", "Semantic embeddings", "Document chunking", "Multi-hop reasoning"],
        "documents_indexed": VECTOR_COLLECTION.count(),
        "embedding_model": "all-MiniLM-L6-v2"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model": "llama-3.3-70b-versatile",
        "vector_db": "ChromaDB",
        "embedding_model": "all-MiniLM-L6-v2",
        "documents_indexed": VECTOR_COLLECTION.count(),
        "vector_rag": True
    }

@app.get("/documents")
def list_documents():
    try:
        # Get unique sources from collection
        all_results = VECTOR_COLLECTION.get()
        sources = {}
        
        for metadata in all_results['metadatas']:
            source = metadata['source']
            if source not in sources:
                sources[source] = {
                    "filename": source,
                    "type": metadata['type'],
                    "chunks": 0
                }
            sources[source]["chunks"] += 1
        
        return {
            "total_documents": len(sources),
            "total_chunks": len(all_results['metadatas']),
            "documents": list(sources.values())
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/reindex")
def reindex_documents():
    """Reindex all documents in the vector database"""
    try:
        # Delete existing collection
        chroma_client.delete_collection("documents")
        
        # Reinitialize
        global VECTOR_COLLECTION
        VECTOR_COLLECTION = initialize_vector_db()
        
        return {
            "message": "Documents reindexed successfully",
            "documents_indexed": VECTOR_COLLECTION.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)