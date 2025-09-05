#!/bin/bash

echo "🚀 Starting Advanced RAG Production System..."

# Start Redis (if not running)
if ! pgrep -x "redis-server" > /dev/null; then
    echo "Starting Redis..."
    redis-server --daemonize yes
fi

# Start optimized RAG API
echo "Starting Optimized RAG API on port 8010..."
python -m uvicorn optimized_rag:app --host 0.0.0.0 --port 8010 &

# Wait for API to start
sleep 5

# Start enhanced Streamlit app
echo "Starting Enhanced Streamlit UI..."
streamlit run enhanced_streamlit_app.py --server.port 8501

echo "✅ Production system started!"
echo "🌐 Access at: http://localhost:8501"