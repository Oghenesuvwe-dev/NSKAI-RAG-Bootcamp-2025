from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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
from typing import Dict, Any, List
import time
from concurrent.futures import ThreadPoolExecutor
import aiohttp

load_dotenv()

# Configure logging
logger.add("logs/optimized_rag.log", rotation="1 day", retention="7 days", level="INFO")

# Rate limiting with higher limits for production
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Optimized RAG Agent", version="10.0.0")
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

# Initialize components with optimization
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./optimized_chroma_db")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Redis cache with connection pooling
try:
    cache_pool = redis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=10)
    cache = redis.Redis(connection_pool=cache_pool, decode_responses=True)
    cache.ping()
    CACHE_ENABLED = True
    logger.info("Redis cache pool initialized")
except:
    CACHE_ENABLED = False
    logger.warning("Redis cache not available")

class QueryRequest(BaseModel):
    query: str
    max_sources: int = 3
    enable_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    reasoning_steps: list
    sources: list
    retrieved_documents: list
    processing_time: float
    cache_hit: bool = False

# Async cache operations
async def get_cached_result(query: str) -> Dict[str, Any]:
    if not CACHE_ENABLED:
        return None
    
    try:
        cache_key = f"opt_rag:{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit: {query[:50]}...")
            return json.loads(cached_result)
    except Exception as e:
        logger.error(f"Cache read error: {e}")
    
    return None

async def cache_result(query: str, result: Dict[str, Any]):
    if not CACHE_ENABLED:
        return
    
    try:
        cache_key = f"opt_rag:{hashlib.md5(query.encode()).hexdigest()}"
        cache.setex(cache_key, 7200, json.dumps(result))  # 2 hour TTL
        logger.info(f"Cached result: {query[:50]}...")
    except Exception as e:
        logger.error(f"Cache write error: {e}")

def initialize_optimized_db():
    try:
        collection = chroma_client.get_collection("optimized_documents")
        logger.info(f"Loaded collection: {collection.count()} documents")
        return collection
    except:
        logger.info("Creating optimized collection")
        collection = chroma_client.create_collection("optimized_documents")
        
        data_dir = Path("data")
        if not data_dir.exists():
            return collection
            
        documents, metadatas, ids = [], [], []
        
        # Parallel document processing
        def process_file(file_path):
            try:
                doc_data = DocumentProcessor.process_document(file_path)
                if doc_data["metadata"].get("error"):
                    return None
                
                # Optimized chunking
                content = doc_data["content"]
                chunk_size = 400
                overlap = 50
                chunks = []
                
                for i in range(0, len(content), chunk_size - overlap):
                    chunk = content[i:i + chunk_size]
                    if len(chunk.strip()) > 50:  # Skip very short chunks
                        chunks.append(chunk)
                
                return file_path, doc_data, chunks
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return None
        
        # Process files in parallel
        file_paths = list(data_dir.glob("*"))
        results = list(executor.map(process_file, [f for f in file_paths if f.is_file()]))
        
        for result in results:
            if result:
                file_path, doc_data, chunks = result
                for i, chunk in enumerate(chunks):
                    doc_id = hashlib.md5(f"{file_path.name}_{i}".encode()).hexdigest()
                    documents.append(chunk)
                    
                    metadata = doc_data["metadata"].copy()
                    metadata.update({
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                        "optimized": True
                    })
                    
                    metadatas.append(metadata)
                    ids.append(doc_id)
        
        if documents:
            # Batch insert for better performance
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_metas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )
            
            logger.info(f"Indexed {len(documents)} optimized chunks")
        
        return collection

OPTIMIZED_COLLECTION = initialize_optimized_db()

async def parallel_retrieval(sub_questions: List[str], max_sources: int = 3):
    """Parallel retrieval for multiple sub-questions"""
    
    def retrieve_for_question(sub_q):
        try:
            results = OPTIMIZED_COLLECTION.query(
                query_texts=[sub_q],
                n_results=max_sources
            )
            
            if results['documents'] and results['documents'][0]:
                return {
                    "sub_question": sub_q,
                    "results": [
                        {
                            "content": results['documents'][0][i],
                            "source": results['metadatas'][0][i]['source'],
                            "similarity_score": 1 - results['distances'][0][i],
                            "metadata": results['metadatas'][0][i]
                        }
                        for i in range(len(results['documents'][0]))
                    ]
                }
            return {"sub_question": sub_q, "results": []}
        except Exception as e:
            logger.error(f"Retrieval error for '{sub_q}': {e}")
            return {"sub_question": sub_q, "results": []}
    
    # Execute retrievals in parallel
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, retrieve_for_question, sub_q)
        for sub_q in sub_questions
    ]
    
    return await asyncio.gather(*tasks)

@app.post("/query")
@limiter.limit("20/minute")  # Increased rate limit for optimized version
async def optimized_query(request: Request, query_request: QueryRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    cache_hit = False
    
    try:
        logger.info(f"Processing optimized query: {query_request.query[:100]}...")
        
        # Check cache if enabled
        if query_request.enable_cache:
            cached_result = await get_cached_result(query_request.query)
            if cached_result:
                cached_result["cache_hit"] = True
                return QueryResponse(**cached_result)
        
        # Optimized query decomposition
        decomp_response = client.chat.completions.create(
            messages=[{
                "role": "user", 
                "content": f'Break down into 2-3 focused sub-questions: "{query_request.query}". Return JSON array only.'
            }],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=200  # Limit tokens for faster response
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [query_request.query]
        
        # Parallel multi-hop retrieval
        retrieval_results = await parallel_retrieval(sub_questions, query_request.max_sources)
        
        # Process results
        retrieved_docs = []
        reasoning_steps = []
        
        for i, result in enumerate(retrieval_results, 1):
            if result["results"]:
                best_result = result["results"][0]  # Take best match
                retrieved_docs.append({
                    "sub_question": result["sub_question"],
                    "content": best_result["content"],
                    "source": best_result["source"],
                    "similarity_score": best_result["similarity_score"]
                })
                reasoning_steps.append(
                    f"Step {i}: {result['sub_question']} → {best_result['source']} "
                    f"(similarity: {best_result['similarity_score']:.3f})"
                )
        
        # Optimized synthesis
        synthesis_prompt = f"""Query: {query_request.query}

Evidence:
{chr(10).join([f"- {doc['source']}: {doc['content'][:200]}..." for doc in retrieved_docs])}

Provide a concise, well-cited answer."""
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1200  # Optimized token limit
        )
        
        processing_time = time.time() - start_time
        sources = list(set([doc["source"] for doc in retrieved_docs]))
        
        result = {
            "answer": final_response.choices[0].message.content,
            "reasoning_steps": reasoning_steps,
            "sources": sources,
            "retrieved_documents": retrieved_docs,
            "processing_time": processing_time,
            "cache_hit": cache_hit
        }
        
        # Cache result in background
        if query_request.enable_cache:
            background_tasks.add_task(cache_result, query_request.query, result)
        
        logger.info(f"Optimized query completed in {processing_time:.2f}s")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Optimized query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "10.0.0",
        "documents": OPTIMIZED_COLLECTION.count(),
        "cache_enabled": CACHE_ENABLED,
        "optimizations": ["parallel_retrieval", "connection_pooling", "batch_processing", "async_caching"],
        "performance": "optimized"
    }

@app.get("/performance")
@limiter.limit("10/minute")
async def performance_metrics(request: Request):
    try:
        metrics = {
            "documents_indexed": OPTIMIZED_COLLECTION.count(),
            "cache_enabled": CACHE_ENABLED,
            "thread_pool_workers": executor._max_workers,
            "optimizations_active": [
                "parallel_processing",
                "connection_pooling", 
                "async_operations",
                "batch_indexing"
            ]
        }
        
        if CACHE_ENABLED:
            cache_info = cache.info()
            metrics["cache_stats"] = {
                "keys": cache.dbsize(),
                "memory_used": cache_info.get("used_memory_human", "N/A"),
                "hits": cache_info.get("keyspace_hits", 0),
                "misses": cache_info.get("keyspace_misses", 0)
            }
        
        return metrics
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        return {"error": "Metrics unavailable"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8010, 
        workers=1,  # Single worker for shared state
        log_level="info"
    )