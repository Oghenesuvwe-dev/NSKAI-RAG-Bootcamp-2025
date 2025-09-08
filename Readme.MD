# 🚀 Advanced Multi-Hop RAG Agent

AI agent that decomposes complex questions into sub-queries, retrieves evidence from multiple real-world data sources, and synthesizes cited answers with interactive visualizations.

## 🎯 **Features**

### **Core Capabilities**
- 🧠 **Multi-hop reasoning** with query decomposition
- 📈 **Real-time Yahoo Finance** stock data
- ⚽ **Sports predictions** (Premier League, World Cup, AFCON)
- 🏢 **Business intelligence** (GenAI market, Big Tech strategies)
- 📊 **Interactive visualizations** with Plotly charts
- 🔄 **Multi-source evidence synthesis**

### **Data Sources**
- **Yahoo Finance** - Real-time stock prices, market data
- **Football Analytics** - Premier League standings, predictions
- **Business Intelligence** - AI market analysis, tech strategies
- **Groq LLM** - llama-3.3-70b for reasoning and synthesis

### **Visualizations**
- ⏱️ Processing time gauge
- 📊 Data sources pie chart
- 📈 Sub-questions complexity analysis
- 📥 Export options (JSON, Markdown, Summary)

## 🚀 **Quick Start**

### **Local Development**
```bash
# Clone and setup
git clone <repository-url>
cd "NSK BOOTCAMP RAG PROJECTS/Advance RAG PROJ"

# Install dependencies
pip install -r requirements.txt

# Add API key to secrets
echo 'GROQ_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Run unified app with visualizations
streamlit run unified_rag_with_graphs.py
```

### **Streamlit Cloud Deployment**
1. Use `unified_rag_with_graphs.py` as main file
2. Add `GROQ_API_KEY` to Streamlit secrets
3. Deploy directly (no backend needed)

## 📊 **Example Queries**

### **💰 Finance & Markets**
- "How did Microsoft's $13B OpenAI investment affect Google's stock price?"
- "What's the impact of GenAI adoption on NVIDIA and tech valuations?"

### **⚽ Sports & Competitions**
- "Based on current standings, who will win the Premier League?"
- "Which team has the best chance to win the 2026 FIFA World Cup?"

### **🏢 Business Intelligence**
- "How is the $148B generative AI market transforming tech economics?"
- "Compare Microsoft, Google, and Amazon's AI strategies and market impact"

## 🏗️ **Architecture**

```
User Query → Decomposition → Multi-Hop Retrieval → Synthesis → Visualized Results
     │            │               │                │              │
   Groq LLM   Sub-queries   Real-time APIs    Final Response  Plotly Charts
```

## 📁 **Project Structure**

```
├── unified_rag_with_graphs.py    # Main application (recommended)
├── market_data_rag.py           # Backend API (optional)
├── enhanced_market_app.py       # Frontend UI (optional)
├── requirements.txt             # Dependencies
├── .streamlit/secrets.toml      # API keys
├── data/                        # Sample documents
└── rag-systems/                 # Alternative implementations
```

## 🔧 **System Requirements**

- Python 3.8+
- Groq API key
- Internet connection (for real-time data)
- 2GB RAM minimum

## 🎯 **Roadmap & Improvements**

### **Immediate Priorities**
- [ ] Real Sports API integration
- [ ] Interactive stock charts
- [ ] Query history system
- [ ] Enhanced error handling

### **Short-term Goals**
- [ ] Multi-model LLM support
- [ ] Advanced visualizations
- [ ] Voice input capability
- [ ] Mobile optimization

### **Long-term Vision**
- [ ] Predictive analytics
- [ ] Enterprise features
- [ ] Custom data sources
- [ ] API development

## 📈 **Performance**

- **Processing Time:** 2-5 seconds average
- **Data Sources:** 3+ real-time APIs
- **Accuracy:** Multi-hop verification
- **Scalability:** Single-file deployment

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch
3. Add improvements from TODO.md
4. Submit pull request

## 📄 **License**

MIT License - NSKAI Bootcamp 2025

## 🏆 **NSKAI Hackathon Submission**

**Repository:** https://github.com/Oghenesuvwe-dev/NSKAI-RAG-Bootcamp-2025  
**Branch:** Advanced Multi-Hop RAG Agent  
**Demo:** Streamlit Cloud deployment ready