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
from document_processor import DocumentProcessor

load_dotenv()

app = FastAPI(title="Enhanced Vector RAG Agent", version="8.0.0")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize ChromaDB and embedding model
chroma_client = chromadb.PersistentClient(path="./enhanced_chroma_db")
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

def initialize_enhanced_vector_db():
    """Initialize ChromaDB with enhanced document processing"""
    try:
        collection = chroma_client.get_collection("enhanced_documents")
        return collection
    except:
        collection = chroma_client.create_collection("enhanced_documents")
        
        # Load and process documents with enhanced processor
        data_dir = Path("data")
        if not data_dir.exists():
            return collection
            
        documents = []
        metadatas = []
        ids = []
        
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                try:
                    # Use enhanced document processor
                    doc_data = DocumentProcessor.process_document(file_path)
                    
                    if doc_data["metadata"].get("error"):
                        continue
                    
                    content = doc_data["content"]
                    chunks = chunk_document(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc_id = hashlib.md5(f"{file_path.name}_{i}".encode()).hexdigest()
                        documents.append(chunk)
                        
                        # Enhanced metadata
                        metadata = doc_data["metadata"].copy()
                        metadata.update({
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "processing_method": "enhanced"
                        })
                        
                        metadatas.append(metadata)
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

# Initialize enhanced vector database
ENHANCED_COLLECTION = initialize_enhanced_vector_db()

def enhanced_vector_search(query: str, top_k: int = 3):
    """Enhanced semantic vector search"""
    try:
        results = ENHANCED_COLLECTION.query(
            query_texts=[query],
            n_results=top_k
        )
        
        search_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                search_results.append({
                    "content": results['documents'][0][i],
                    "metadata": metadata,
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                    "source": metadata['source'],
                    "doc_type": metadata.get('type', 'unknown')
                })
        
        return search_results
    except Exception as e:
        print(f"Enhanced vector search error: {e}")
        return []

def multi_hop_retrieval(sub_query: str, main_query: str):
    """Enhanced multi-hop retrieval with document type awareness"""
    search_query = f"{sub_query} {main_query}"
    results = enhanced_vector_search(search_query, top_k=1)
    
    if results:
        best_match = results[0]
        metadata = best_match["metadata"]
        
        return {
            "sub_question": sub_query,
            "document": best_match["source"],
            "document_type": best_match["doc_type"],
            "similarity_score": 1 - best_match["distance"],
            "content": best_match["content"],
            "source": best_match["source"],
            "chunk_info": f"Chunk {metadata['chunk_id'] + 1}/{metadata['total_chunks']}",
            "metadata": {k: v for k, v in metadata.items() if k not in ['chunk_id', 'total_chunks']}
        }
    else:
        return {
            "sub_question": sub_query,
            "document": "No relevant document found",
            "document_type": "none",
            "similarity_score": 0,
            "content": "No matching content found for this query.",
            "source": "None",
            "chunk_info": "N/A",
            "metadata": {}
        }

@app.post("/query", response_model=QueryResponse)
def run_enhanced_query(request: QueryRequest):
    try:
        collection_count = ENHANCED_COLLECTION.count()
        if collection_count == 0:
            raise HTTPException(status_code=500, detail="No documents in enhanced vector database.")
        
        # Query decomposition
        decomposition_prompt = f"""
        Decompose this complex query into 2-4 specific sub-questions for multi-hop reasoning:
        Query: "{request.query}"
        
        Return ONLY a JSON array of sub-questions:
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
            sub_questions = [request.query]
        
        # Enhanced multi-hop retrieval
        retrieved_docs = []
        reasoning_steps = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            doc_result = multi_hop_retrieval(sub_q, request.query)
            retrieved_docs.append(doc_result)
            reasoning_steps.append(f"Step {i}: {sub_q} → {doc_result['document_type'].upper()} file: {doc_result['source']} (similarity: {doc_result['similarity_score']:.3f})")
        
        # Enhanced synthesis with document type awareness
        synthesis_prompt = f"""
        Synthesize information from multiple document types to answer this query:
        
        Original Query: {request.query}
        
        Retrieved Evidence from Different Document Types:
        {json.dumps([{k: v for k, v in doc.items() if k != 'content'} for doc in retrieved_docs], indent=2)}
        
        Document Contents:
        {chr(10).join([f"[{doc['document_type'].upper()}] {doc['source']} ({doc['chunk_info']}){chr(10)}{doc['content']}{chr(10)}" for doc in retrieved_docs])}
        
        Provide a comprehensive answer that:
        1. Addresses the original query directly
        2. Uses specific data from different document types (PDF, CSV, etc.)
        3. Shows reasoning connecting evidence across document types
        4. Cites sources with document type information
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
        "message": "Enhanced Vector RAG Multi-Hop Research Agent", 
        "version": "8.0.0",
        "features": ["Multi-format document processing", "PDF/CSV/Excel support", "Enhanced metadata", "ChromaDB vector search"],
        "documents_indexed": ENHANCED_COLLECTION.count()
    }

@app.get("/health")
def health():
    return {
        "status": "healthy", 
        "model": "llama-3.3-70b-versatile",
        "vector_db": "ChromaDB Enhanced",
        "document_processor": "Multi-format",
        "documents_indexed": ENHANCED_COLLECTION.count(),
        "enhanced_rag": True
    }

@app.get("/documents")
def list_enhanced_documents():
    try:
        all_results = ENHANCED_COLLECTION.get()
        sources = {}
        
        for metadata in all_results['metadatas']:
            source = metadata['source']
            if source not in sources:
                sources[source] = {
                    "filename": source,
                    "type": metadata['type'],
                    "chunks": 0,
                    "processing_method": metadata.get('processing_method', 'basic')
                }
            sources[source]["chunks"] += 1
        
        return {
            "total_documents": len(sources),
            "total_chunks": len(all_results['metadatas']),
            "documents": list(sources.values()),
            "supported_formats": ["PDF", "CSV", "Excel", "JSON", "Markdown", "Text"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/reindex")
def reindex_enhanced_documents():
    """Reindex with enhanced document processing"""
    try:
        chroma_client.delete_collection("enhanced_documents")
        global ENHANCED_COLLECTION
        ENHANCED_COLLECTION = initialize_enhanced_vector_db()
        
        return {
            "message": "Enhanced documents reindexed successfully",
            "documents_indexed": ENHANCED_COLLECTION.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)