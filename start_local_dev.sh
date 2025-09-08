#!/bin/bash

echo "🚀 Starting Local Development Environment..."

# Kill existing processes
pkill -f "production_rag"
pkill -f "enhanced_streamlit_app"

# Start backend
echo "Starting Production RAG API on port 8009..."
python3 -m uvicorn production_rag:app --port 8009 &
BACKEND_PID=$!

# Wait for backend
sleep 3

# Start frontend
echo "Starting Streamlit UI on port 8501..."
streamlit run enhanced_streamlit_app.py --server.port 8501 &
FRONTEND_PID=$!

echo "✅ Local development started!"
echo "🌐 Streamlit: http://localhost:8501"
echo "📡 API: http://localhost:8009"

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait