import yfinance as yf
import requests
from datetime import datetime
from typing import Dict

class DataSources:
    def get_data(self, query: str, source: str) -> Dict:
        if source == "finance":
            return self.get_stock_data(query)
        elif source == "sports":
            return self.get_sports_data(query)
        elif source == "business":
            return self.get_business_data(query)
        return {"data": "General information"}
    
    def get_stock_data(self, query: str) -> Dict:
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN"]
        for symbol in symbols:
            if symbol.lower() in query.lower():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="5d")
                    info = ticker.info
                    
                    return {
                        "symbol": symbol,
                        "current_price": round(hist['Close'].iloc[-1], 2),
                        "change_pct": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100, 2),
                        "market_cap": info.get('marketCap', 'N/A'),
                        "company_name": info.get('longName', symbol)
                    }
                except:
                    pass
        
        return {"data": "Stock market analysis", "symbols": symbols}
    
    def get_sports_data(self, query: str) -> Dict:
        if "premier" in query.lower():
            try:
                url = "https://www.thesportsdb.com/api/v1/json/3/lookuptable.php?l=4328&s=2024-2025"
                response = requests.get(url, timeout=5)
                data = response.json()
                
                return {
                    "competition": "Premier League",
                    "standings": data.get('table', [])[:6],
                    "last_updated": datetime.now().isoformat()
                }
            except:
                pass
        
        elif "nba" in query.lower():
            return {
                "competition": "NBA",
                "top_teams": ["Boston Celtics", "Oklahoma City Thunder", "Cleveland Cavaliers"],
                "season": "2024-25"
            }
        
        return {"data": "Sports analysis", "leagues": ["Premier League", "NBA"]}
    
    def get_business_data(self, query: str) -> Dict:
        business_topics = {
            "microsoft ai": {"investment": "$13B in OpenAI", "strategy": "Azure AI integration", "impact": "35% Azure growth"},
            "google ai": {"investment": "$70B+ in AI R&D", "strategy": "Bard & Search integration", "impact": "Search dominance"},
            "ai market": {"size": "$148B by 2030", "growth": "36.8% CAGR", "leaders": "Microsoft, Google, OpenAI"},
            "tech valuations": {"sector": "Technology", "trend": "AI-driven growth", "multiples": "25-30x P/E"}
        }
        
        for topic, data in business_topics.items():
            if any(word in query.lower() for word in topic.split()):
                return {"topic": topic, "analysis": data}
        
        return {"topic": "general business", "analysis": "Market trends and competitive analysis"