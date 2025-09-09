# 🚀 Quick Setup Guide

## 1. Get Groq API Key
1. Go to [https://console.groq.com/keys](https://console.groq.com/keys)
2. Sign up/login
3. Create new API key
4. Copy the key

## 2. Set API Key
**Option A: Environment Variable**
```bash
export GROQ_API_KEY="your_actual_key_here"
```

**Option B: Streamlit Secrets**
Edit `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "your_actual_key_here"
```

## 3. Run App
```bash
python3 -m streamlit run app.py
```

## 4. Test Queries
- "Compare Microsoft's AI investment impact on stock vs Google's strategy"
- "How did Liverpool's performance affect Champions League chances vs Arsenal?"
- "What's the correlation between tech stocks and sports team valuations?"