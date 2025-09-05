from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
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
from loguru import logger
import redis
import asyncio
from typing import Dict, Any
import time

load_dotenv()

# Configure logging
logger.add("logs/rag_app.log", rotation="1 day", retention="7 days", level="INFO")

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Production RAG Agent", version="9.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./production_chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Redis cache (optional)
try:
    cache = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
    logger.info("Redis cache connected")
except:
    CACHE_ENABLED = False
    logger.warning("Redis cache not available")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    reasoning_steps: list
    sources: list
    retrieved_documents: list
    processing_time: float

def get_cache_key(query: str) -> str:
    return f"rag_query:{hashlib.md5(query.encode()).hexdigest()}"

async def cached_query(query: str) -> Dict[str, Any]:
    if not CACHE_ENABLED:
        return None
    
    try:
        cache_key = get_cache_key(query)
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for query: {query[:50]}...")
            return json.loads(cached_result)
    except Exception as e:
        logger.error(f"Cache read error: {e}")
    
    return None

async def cache_result(query: str, result: Dict[str, Any]):
    if not CACHE_ENABLED:
        return
    
    try:
        cache_key = get_cache_key(query)
        cache.setex(cache_key, 3600, json.dumps(result))  # 1 hour TTL
        logger.info(f"Cached result for query: {query[:50]}...")
    except Exception as e:
        logger.error(f"Cache write error: {e}")

def initialize_production_db():
    try:
        collection = chroma_client.get_collection("production_documents")
        logger.info(f"Loaded existing collection with {collection.count()} documents")
        return collection
    except:
        logger.info("Creating new document collection")
        collection = chroma_client.create_collection("production_documents")
        
        data_dir = Path("data")
        if not data_dir.exists():
            logger.warning("No data directory found")
            return collection
            
        documents, metadatas, ids = [], [], []
        
        for file_path in data_dir.glob("*"):
            if file_path.is_file():
                try:
                    doc_data = DocumentProcessor.process_document(file_path)
                    if doc_data["metadata"].get("error"):
                        continue
                    
                    chunks = [doc_data["content"][i:i+500] for i in range(0, len(doc_data["content"]), 450)]
                    
                    for i, chunk in enumerate(chunks):
                        doc_id = hashlib.md5(f"{file_path.name}_{i}".encode()).hexdigest()
                        documents.append(chunk)
                        metadata = doc_data["metadata"].copy()
                        metadata.update({"chunk_id": i, "total_chunks": len(chunks)})
                        metadatas.append(metadata)
                        ids.append(doc_id)
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info(f"Indexed {len(documents)} document chunks")
        
        return collection

PRODUCTION_COLLECTION = initialize_production_db()

@app.post("/query")
@limiter.limit("10/minute")
async def production_query(request: Request, query_request: QueryRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {query_request.query[:100]}...")
        
        # Check cache first
        cached_result = await cached_query(query_request.query)
        if cached_result:
            return QueryResponse(**cached_result)
        
        # Query decomposition
        decomp_response = client.chat.completions.create(
            messages=[{"role": "user", "content": f'Decompose into 2-3 sub-questions: "{query_request.query}". Return JSON array only.'}],
            model="llama-3.3-70b-versatile",
            temperature=0.1
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [query_request.query]
        
        # Multi-hop retrieval
        retrieved_docs = []
        reasoning_steps = []
        
        for i, sub_q in enumerate(sub_questions, 1):
            results = PRODUCTION_COLLECTION.query(query_texts=[sub_q], n_results=1)
            
            if results['documents'] and results['documents'][0]:
                doc = {
                    "sub_question": sub_q,
                    "content": results['documents'][0][0],
                    "source": results['metadatas'][0][0]['source'],
                    "similarity_score": 1 - results['distances'][0][0]
                }
                retrieved_docs.append(doc)
                reasoning_steps.append(f"Step {i}: {sub_q} → {doc['source']} (score: {doc['similarity_score']:.3f})")
        
        # Synthesis
        synthesis_prompt = f"""Answer: {query_request.query}

Evidence:
{chr(10).join([f"{doc['source']}: {doc['content']}" for doc in retrieved_docs])}

Provide a comprehensive answer with citations."""
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1500
        )
        
        processing_time = time.time() - start_time
        sources = list(set([doc["source"] for doc in retrieved_docs]))
        
        result = {
            "answer": final_response.choices[0].message.content,
            "reasoning_steps": reasoning_steps,
            "sources": sources,
            "retrieved_documents": retrieved_docs,
            "processing_time": processing_time
        }
        
        # Cache result
        await cache_result(query_request.query, result)
        
        logger.info(f"Query completed in {processing_time:.2f}s")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "9.0.0",
        "documents": PRODUCTION_COLLECTION.count(),
        "cache_enabled": CACHE_ENABLED,
        "features": ["rate_limiting", "caching", "logging", "error_handling"]
    }

@app.get("/metrics")
@limiter.limit("5/minute")
async def get_metrics(request: Request):
    try:
        cache_info = {"enabled": CACHE_ENABLED}
        if CACHE_ENABLED:
            cache_info.update({
                "keys": cache.dbsize(),
                "memory": cache.info().get("used_memory_human", "N/A")
            })
        
        return {
            "documents_indexed": PRODUCTION_COLLECTION.count(),
            "cache": cache_info,
            "uptime": "Available via /health"
        }
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        return {"error": "Metrics unavailable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8009, log_level="info")