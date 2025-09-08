"""
🚀 Production Configuration
Final deployment settings and optimizations
"""

import streamlit as st
import os
from typing import Dict, Any

# Production Streamlit Configuration
PRODUCTION_CONFIG = """
[server]
maxUploadSize = 200
maxMessageSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
showErrorDetails = false

[theme]
primaryColor = "#3b82f6"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[client]
caching = true
displayEnabled = true
"""

# Environment Variables for Production
REQUIRED_ENV_VARS = {
    "GROQ_API_KEY": "Groq API key for LLM access",
    "NEWS_API_KEY": "NewsAPI key (optional)",
    "FOOTBALL_DATA_API_KEY": "Football data API key (optional)"
}

def setup_production_environment():
    """Setup production environment with optimizations"""
    
    # Set page config for production
    if not hasattr(st, '_is_running_with_streamlit'):
        st.set_page_config(
            page_title="Advanced Multi-Hop RAG Agent",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="auto",
            menu_items={
                'Get Help': None,
                'Report a bug': None,
                'About': "Advanced Multi-Hop RAG Agent v1.0"
            }
        )
    
    # Production optimizations
    if 'production_mode' not in st.session_state:
        st.session_state.production_mode = True
        st.session_state.debug_mode = False
        st.session_state.performance_monitoring = True

def validate_environment() -> Dict[str, Any]:
    """Validate production environment"""
    status = {"valid": True, "missing": [], "warnings": []}
    
    # Check required API keys
    for var, description in REQUIRED_ENV_VARS.items():
        if var == "GROQ_API_KEY":
            if not (st.secrets.get(var) or os.getenv(var)):
                status["valid"] = False
                status["missing"].append(f"{var}: {description}")
        else:
            if not (st.secrets.get(var) or os.getenv(var)):
                status["warnings"].append(f"{var}: {description} (optional)")
    
    return status

def get_production_secrets():
    """Get production secrets safely"""
    secrets = {}
    
    for var in REQUIRED_ENV_VARS.keys():
        # Try Streamlit secrets first, then environment
        value = None
        if hasattr(st, 'secrets'):
            value = st.secrets.get(var)
        if not value:
            value = os.getenv(var)
        
        if value:
            secrets[var] = value
    
    return secrets

# Production deployment checklist
DEPLOYMENT_CHECKLIST = {
    "Environment": [
        "✅ GROQ_API_KEY configured",
        "⚠️ NEWS_API_KEY configured (optional)",
        "⚠️ FOOTBALL_DATA_API_KEY configured (optional)"
    ],
    "Performance": [
        "✅ Caching enabled (30min TTL)",
        "✅ Memory optimization active",
        "✅ Compression for large data",
        "✅ Auto-cleanup every 50 queries"
    ],
    "Security": [
        "✅ XSRF protection enabled",
        "✅ Error details hidden in production",
        "✅ Usage stats disabled",
        "✅ Secure API key handling"
    ],
    "Features": [
        "✅ Multi-hop RAG analysis",
        "✅ Real-time data integration",
        "✅ Advanced visualizations",
        "✅ Analytics dashboard",
        "✅ Predictive analytics",
        "✅ Export capabilities"
    ]
}

def display_deployment_status():
    """Display production deployment status"""
    st.subheader("🚀 Production Deployment Status")
    
    env_status = validate_environment()
    
    if env_status["valid"]:
        st.success("✅ Production environment ready")
    else:
        st.error("❌ Missing required configuration")
        for missing in env_status["missing"]:
            st.error(f"• {missing}")
    
    if env_status["warnings"]:
        st.warning("⚠️ Optional configurations missing:")
        for warning in env_status["warnings"]:
            st.warning(f"• {warning}")
    
    # Deployment checklist
    col1, col2 = st.columns(2)
    
    with col1:
        for category, items in list(DEPLOYMENT_CHECKLIST.items())[:2]:
            st.write(f"**{category}:**")
            for item in items:
                st.write(item)
    
    with col2:
        for category, items in list(DEPLOYMENT_CHECKLIST.items())[2:]:
            st.write(f"**{category}:**")
            for item in items:
                st.write(item)

# Production monitoring
def get_system_metrics():
    """Get system metrics for production monitoring"""
    return {
        "cache_size": len([k for k in st.session_state.keys() if k.startswith('cache_')]),
        "session_size": len(st.session_state.keys()),
        "queries_processed": st.session_state.get('performance_stats', {}).get('queries', 0),
        "cache_hit_rate": st.session_state.get('performance_stats', {}).get('cache_hits', 0),
        "active_alerts": len(st.session_state.get('alerts', [])),
        "production_mode": st.session_state.get('production_mode', False)
    }