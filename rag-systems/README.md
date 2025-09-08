# 🚀 Advanced RAG Systems Overview

## 📁 System Organization

### 1. **Production RAG** (Recommended) - `production-rag/`
- **Backend:** `optimized_rag.py` (Port 8010)
- **Frontend:** `enhanced_streamlit_app.py`
- **Features:**
  - ✅ Redis caching (2hr TTL)
  - ✅ Rate limiting (20/min)
  - ✅ Connection pooling
  - ✅ Parallel processing
  - ✅ Performance metrics
  - ✅ Advanced UI with bookmarks

### 2. **Real RAG** - `real-rag/` ⭐ **HAS "API OFFLINE" MESSAGE**
- **Backend:** `real_rag_engine.py` (Port 8006)
- **Frontend:** `final_streamlit_app.py`
- **Features:**
  - 📄 Real document processing from `/data` folder
  - 🔍 Semantic search with keyword matching
  - 🧠 Multi-hop reasoning
  - ❌ Shows "API Disconnected" when offline

### 3. **Market RAG** - `market-rag/`
- **Backend:** `market_data_rag.py` (Port 8004)
- **Frontend:** `market_streamlit_app.py`
- **Features:**
  - 📈 Real-time Yahoo Finance data
  - 🏈 LiveScore sports API
  - 💼 Market news integration
  - 📊 Stock analysis with real data

## 🎯 Quick Start Commands

```bash
# Production RAG (Recommended)
python -m uvicorn optimized_rag:app --port 8010 & streamlit run enhanced_streamlit_app.py

# Real RAG (Has API offline message)
python -m uvicorn real_rag_engine:app --port 8006 & streamlit run final_streamlit_app.py

# Market Data RAG
python -m uvicorn market_data_rag:app --port 8004 & streamlit run market_streamlit_app.py
```

## 🔍 System Identification

**You're currently running:** **Real RAG** - the one with "❌ API Disconnected" message in the UI.

This system processes real documents from the `/data` folder and shows API status messages.