#!/bin/bash

echo "🚀 Starting Enhanced Multi-Hop RAG System..."

# Kill any existing processes
pkill -f "market_data_rag"
pkill -f "enhanced_market_app"

# Start enhanced backend
echo "Starting Enhanced Market RAG API on port 8004..."
python3 -m uvicorn market_data_rag:app --host 0.0.0.0 --port 8004 &

# Wait for API to start
sleep 3

# Start enhanced Streamlit app
echo "Starting Enhanced Multi-Hop UI..."
streamlit run enhanced_market_app.py --server.port 8501

echo "✅ Enhanced system started!"
echo "🌐 Access at: http://localhost:8501"
echo "📡 API at: http://localhost:8004"