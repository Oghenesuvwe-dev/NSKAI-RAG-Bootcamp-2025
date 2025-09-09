# 🎯 Multi-Hop RAG Agent

Clean, focused multi-hop RAG system for Sports, Business, and Finance analysis using live data.

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set API key**
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

## 🎯 Features

- **Multi-hop reasoning** - Breaks complex queries into sub-questions
- **Live data** - Yahoo Finance & TheSportsDB APIs
- **Clean UI** - Simple, focused interface
- **Smart routing** - Intelligent source selection
- **Citations** - Proper source attribution

## 📊 Example Queries

- "How did Liverpool's performance affect Champions League chances vs Arsenal?"
- "Compare Microsoft's AI investment impact vs Google's strategy"
- "What's the correlation between tech stocks and sports valuations?"

## 🛠️ Architecture

```
core/
├── multi_hop_engine.py    # Main RAG logic
├── data_sources.py        # Live APIs
├── knowledge_base.py      # Document storage
└── answer_formatter.py    # Response formatting
```

**Total implementation: 6 tasks, 22 hours → Complete focused multi-hop RAG system**