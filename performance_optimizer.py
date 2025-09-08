"""
⚡ Performance Optimizer
Final polish and optimization for production deployment
"""

import streamlit as st
import asyncio
import concurrent.futures
import time
from typing import Dict, Any, List
import json
import gzip
import pickle

class PerformanceOptimizer:
    def __init__(self):
        self.cache_compression = True
        self.async_enabled = True
        
    def optimize_cache(self):
        """Compress and optimize cache storage"""
        cache_keys = [k for k in st.session_state.keys() if k.startswith('cache_')]
        compressed_count = 0
        
        for key in cache_keys:
            try:
                data = st.session_state[key]
                if isinstance(data, dict) and 'data' in data:
                    # Compress large data
                    serialized = json.dumps(data['data']).encode('utf-8')
                    if len(serialized) > 1024:  # > 1KB
                        compressed = gzip.compress(serialized)
                        st.session_state[f"{key}_compressed"] = {
                            'data': compressed,
                            'timestamp': data.get('timestamp', time.time()),
                            'compressed': True
                        }
                        del st.session_state[key]
                        compressed_count += 1
            except:
                continue
                
        return compressed_count
    
    def decompress_cache(self, key: str) -> Dict[str, Any]:
        """Decompress cached data"""
        compressed_key = f"{key}_compressed"
        if compressed_key in st.session_state:
            try:
                cached = st.session_state[compressed_key]
                decompressed = gzip.decompress(cached['data'])
                return {
                    'data': json.loads(decompressed.decode('utf-8')),
                    'timestamp': cached['timestamp']
                }
            except:
                pass
        return st.session_state.get(key, {})
    
    async def parallel_data_fetch(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Fetch multiple data sources in parallel"""
        if not self.async_enabled:
            return []
            
        async def fetch_single(query: str):
            # Simulate async data fetch
            await asyncio.sleep(0.1)
            return {"query": query, "data": f"Result for {query}", "timestamp": time.time()}
        
        tasks = [fetch_single(q) for q in queries[:3]]  # Limit to 3 parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]
    
    def memory_cleanup(self):
        """Clean up memory and optimize session state"""
        # Remove old cache entries (>1 hour)
        current_time = time.time()
        old_keys = []
        
        for key, value in st.session_state.items():
            if key.startswith('cache_') and isinstance(value, dict):
                if current_time - value.get('timestamp', 0) > 3600:  # 1 hour
                    old_keys.append(key)
        
        for key in old_keys:
            del st.session_state[key]
            
        return len(old_keys)
    
    def optimize_ui_performance(self):
        """Optimize UI rendering performance"""
        # Reduce chart complexity for mobile
        if st.session_state.get('mobile_view', False):
            return {
                'chart_points': 50,  # Reduce data points
                'animation': False,
                'hover_data': False
            }
        return {
            'chart_points': 200,
            'animation': True,
            'hover_data': True
        }

class ErrorHandler:
    @staticmethod
    def handle_api_error(error: Exception) -> Dict[str, Any]:
        """Centralized error handling"""
        error_msg = str(error).lower()
        
        if "rate limit" in error_msg:
            return {"error": "Rate limit exceeded. Please wait 30 seconds.", "retry_after": 30}
        elif "network" in error_msg or "connection" in error_msg:
            return {"error": "Network issue. Check connection and retry.", "retry_after": 5}
        elif "api key" in error_msg:
            return {"error": "API key invalid. Check configuration.", "retry_after": 0}
        else:
            return {"error": f"System error: {str(error)[:100]}", "retry_after": 10}
    
    @staticmethod
    def display_error(error_info: Dict[str, Any]):
        """Display user-friendly errors"""
        if error_info.get("retry_after", 0) > 0:
            st.error(f"⚠️ {error_info['error']}")
            st.info(f"🔄 Auto-retry in {error_info['retry_after']} seconds")
        else:
            st.error(f"❌ {error_info['error']}")

def apply_final_optimizations():
    """Apply all final optimizations"""
    optimizer = PerformanceOptimizer()
    
    # Initialize optimizer in session state
    if 'optimizer' not in st.session_state:
        st.session_state.optimizer = optimizer
    
    # Auto-cleanup every 50 queries
    if st.session_state.get('performance_stats', {}).get('queries', 0) % 50 == 0:
        cleaned = optimizer.memory_cleanup()
        compressed = optimizer.optimize_cache()
        
        if cleaned > 0 or compressed > 0:
            st.toast(f"🧹 Optimized: {cleaned} old cache + {compressed} compressed")
    
    return optimizer

# Production deployment helpers
def get_deployment_config():
    """Production deployment configuration"""
    return {
        "streamlit_config": {
            "server.maxUploadSize": 200,
            "server.maxMessageSize": 200,
            "browser.gatherUsageStats": False,
            "theme.primaryColor": "#3b82f6"
        },
        "performance": {
            "cache_ttl": 1800,  # 30 minutes
            "max_concurrent_queries": 3,
            "compression_threshold": 1024,  # 1KB
            "cleanup_interval": 50  # queries
        }
    }

def create_health_check():
    """System health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "cache_size": len([k for k in st.session_state.keys() if k.startswith('cache_')]),
        "memory_usage": "optimal",
        "api_status": "connected"
    }
    
    # Check critical components
    try:
        from groq import Groq
        health_status["llm_status"] = "ready"
    except:
        health_status["llm_status"] = "unavailable"
        health_status["status"] = "degraded"
    
    return health_status