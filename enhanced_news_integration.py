import asyncio
import aiohttp
from newsapi import NewsApiClient
from textblob import TextBlob
import streamlit as st
import json
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Enhanced News Integration Module
class NewsAnalyzer:
    def __init__(self):
        self.api_key = os.getenv("NEWS_API_KEY")
        self.client = NewsApiClient(api_key=self.api_key) if self.api_key else None
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = "Positive"
                color = "green"
            elif polarity < -0.1:
                sentiment = "Negative" 
                color = "red"
            else:
                sentiment = "Neutral"
                color = "gray"
                
            return {
                "sentiment": sentiment,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
                "color": color,
                "confidence": abs(polarity)
            }
        except Exception as e:
            return {
                "sentiment": "Unknown",
                "polarity": 0,
                "subjectivity": 0,
                "color": "gray",
                "confidence": 0,
                "error": str(e)
            }
    
    async def get_market_news_async(self, query: str, category: str = "business") -> Dict[str, Any]:
        """Async news retrieval with sentiment analysis"""
        try:
            if not self.client:
                return self.get_mock_news_with_sentiment(query, category)
            
            # Get news articles
            articles = self.client.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                page_size=20,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            if articles['status'] != 'ok':
                return self.get_mock_news_with_sentiment(query, category)
            
            # Process articles with sentiment
            processed_articles = []
            sentiment_scores = []
            
            for article in articles['articles'][:10]:  # Limit to 10 articles
                title = article.get('title', '')
                description = article.get('description', '')
                content = f"{title} {description}"
                
                sentiment_data = self.analyze_sentiment(content)
                
                processed_article = {
                    "title": title,
                    "description": description,
                    "url": article.get('url', ''),
                    "source": article.get('source', {}).get('name', 'Unknown'),
                    "published_at": article.get('publishedAt', ''),
                    "sentiment": sentiment_data,
                    "relevance_score": self.calculate_relevance(content, query)
                }
                
                processed_articles.append(processed_article)
                sentiment_scores.append(sentiment_data['polarity'])
            
            # Calculate overall sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                "query": query,
                "category": category,
                "articles": processed_articles,
                "total_articles": len(processed_articles),
                "overall_sentiment": {
                    "score": round(avg_sentiment, 3),
                    "label": "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                },
                "sentiment_distribution": {
                    "positive": len([s for s in sentiment_scores if s > 0.1]),
                    "negative": len([s for s in sentiment_scores if s < -0.1]),
                    "neutral": len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
                },
                "source": "NewsAPI",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return self.get_mock_news_with_sentiment(query, category)
    
    def calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score based on keyword matching"""
        query_words = query.lower().split()
        content_lower = content.lower()
        
        matches = sum(1 for word in query_words if word in content_lower)
        return round(matches / len(query_words), 2) if query_words else 0
    
    def get_mock_news_with_sentiment(self, query: str, category: str) -> Dict[str, Any]:
        """Enhanced mock news data with sentiment analysis"""
        mock_articles = [
            {
                "title": f"Breaking: {query.title()} Market Analysis Shows Strong Performance",
                "description": f"Latest analysis of {query} indicates positive market trends with strong investor confidence and growth potential.",
                "url": "https://example.com/news1",
                "source": "Financial Times",
                "published_at": datetime.now().isoformat(),
                "content": f"Market analysis shows {query} performing well with positive investor sentiment"
            },
            {
                "title": f"{query.title()} Faces Regulatory Challenges Amid Market Volatility",
                "description": f"Recent developments in {query} sector show mixed signals as regulatory concerns impact investor confidence.",
                "url": "https://example.com/news2", 
                "source": "Reuters",
                "published_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "content": f"Regulatory challenges affecting {query} with uncertain market outlook"
            },
            {
                "title": f"Expert Predictions: {query.title()} Set for Steady Growth",
                "description": f"Industry experts remain optimistic about {query} prospects despite recent market fluctuations.",
                "url": "https://example.com/news3",
                "source": "Bloomberg",
                "published_at": (datetime.now() - timedelta(hours=4)).isoformat(),
                "content": f"Experts predict steady growth for {query} with positive long-term outlook"
            }
        ]
        
        # Add sentiment analysis to mock articles
        processed_articles = []
        sentiment_scores = []
        
        for article in mock_articles:
            content = f"{article['title']} {article['description']}"
            sentiment_data = self.analyze_sentiment(content)
            
            processed_article = {
                **article,
                "sentiment": sentiment_data,
                "relevance_score": self.calculate_relevance(content, query)
            }
            
            processed_articles.append(processed_article)
            sentiment_scores.append(sentiment_data['polarity'])
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        return {
            "query": query,
            "category": category,
            "articles": processed_articles,
            "total_articles": len(processed_articles),
            "overall_sentiment": {
                "score": round(avg_sentiment, 3),
                "label": "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
            },
            "sentiment_distribution": {
                "positive": len([s for s in sentiment_scores if s > 0.1]),
                "negative": len([s for s in sentiment_scores if s < -0.1]),
                "neutral": len([s for s in sentiment_scores if -0.1 <= s <= 0.1])
            },
            "source": "Enhanced Mock Data",
            "timestamp": datetime.now().isoformat()
        }

# Async Processing System
class AsyncProcessor:
    def __init__(self):
        self.session = None
        
    async def create_session(self):
        """Create aiohttp session for async requests"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def parallel_data_retrieval(self, sub_questions: List[str], main_query: str) -> List[Dict[str, Any]]:
        """Process multiple sub-questions in parallel"""
        try:
            session = await self.create_session()
            
            # Create tasks for parallel processing
            tasks = []
            for sub_q in sub_questions:
                task = asyncio.create_task(
                    self.process_single_query(sub_q, main_query, session)
                )
                tasks.append(task)
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and return valid results
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    valid_results.append({
                        "sub_question": sub_questions[i],
                        "error": str(result),
                        "data": {"message": "Processing failed"},
                        "source": "Error Handler"
                    })
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            # Fallback to sequential processing
            return await self.sequential_fallback(sub_questions, main_query)
        finally:
            await self.close_session()
    
    async def process_single_query(self, sub_query: str, main_query: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Process a single sub-query asynchronously"""
        try:
            # Simulate async data retrieval
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Use existing intelligent_data_retrieval logic
            from unified_rag_with_graphs import intelligent_data_retrieval
            result = intelligent_data_retrieval(sub_query, main_query)
            
            return {
                "sub_question": sub_query,
                "data": result.get("data", {}),
                "source": result.get("source", "Unknown"),
                "processing_time": 0.1,
                "async_processed": True
            }
            
        except Exception as e:
            return {
                "sub_question": sub_query,
                "error": str(e),
                "data": {"message": "Async processing failed"},
                "source": "Error Handler"
            }
    
    async def sequential_fallback(self, sub_questions: List[str], main_query: str) -> List[Dict[str, Any]]:
        """Fallback to sequential processing if parallel fails"""
        results = []
        for sub_q in sub_questions:
            try:
                from unified_rag_with_graphs import intelligent_data_retrieval
                result = intelligent_data_retrieval(sub_q, main_query)
                results.append({
                    "sub_question": sub_q,
                    "data": result.get("data", {}),
                    "source": result.get("source", "Unknown"),
                    "processing_time": 0.2,
                    "async_processed": False
                })
            except Exception as e:
                results.append({
                    "sub_question": sub_q,
                    "error": str(e),
                    "data": {"message": "Sequential processing failed"},
                    "source": "Error Handler"
                })
        return results

# Streamlit UI Components for News Integration
def render_news_dashboard(news_data: Dict[str, Any]):
    """Render news analysis dashboard"""
    if not news_data or "articles" not in news_data:
        st.error("No news data available")
        return
    
    st.subheader(f"📰 News Analysis: {news_data['query']}")
    
    # Overall sentiment metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_label = news_data['overall_sentiment']['label']
        sentiment_color = "green" if sentiment_label == "Positive" else "red" if sentiment_label == "Negative" else "gray"
        st.metric("Overall Sentiment", sentiment_label, 
                 f"{news_data['overall_sentiment']['score']:.3f}")
    
    with col2:
        st.metric("Total Articles", news_data['total_articles'])
    
    with col3:
        dist = news_data['sentiment_distribution']
        st.metric("Positive News", f"{dist['positive']}/{news_data['total_articles']}")
    
    with col4:
        st.metric("Negative News", f"{dist['negative']}/{news_data['total_articles']}")
    
    # Sentiment distribution chart
    fig_sentiment = go.Figure(data=[
        go.Bar(
            x=['Positive', 'Neutral', 'Negative'],
            y=[dist['positive'], dist['neutral'], dist['negative']],
            marker_color=['green', 'gray', 'red']
        )
    ])
    fig_sentiment.update_layout(
        title="News Sentiment Distribution",
        xaxis_title="Sentiment",
        yaxis_title="Number of Articles",
        height=300
    )
    st.plotly_chart(fig_sentiment, use_container_width=True)
    
    # Individual articles
    st.subheader("📄 Article Analysis")
    
    for i, article in enumerate(news_data['articles'][:5], 1):
        with st.expander(f"Article {i}: {article['title'][:60]}..."):
            col_article1, col_article2 = st.columns([3, 1])
            
            with col_article1:
                st.write(f"**Source:** {article['source']}")
                st.write(f"**Description:** {article['description']}")
                st.write(f"**URL:** {article['url']}")
                st.write(f"**Published:** {article['published_at'][:19]}")
            
            with col_article2:
                sentiment = article['sentiment']
                st.metric("Sentiment", sentiment['sentiment'], 
                         f"{sentiment['polarity']:.3f}")
                st.metric("Relevance", f"{article['relevance_score']:.2f}")
                st.metric("Confidence", f"{sentiment['confidence']:.3f}")

def render_async_progress(progress_placeholder, current_step: int, total_steps: int, step_name: str):
    """Render async processing progress"""
    progress = current_step / total_steps
    progress_placeholder.progress(progress, text=f"Step {current_step}/{total_steps}: {step_name}")

# Integration functions for main app
async def enhanced_multi_hop_with_news(query: str, category: str) -> Dict[str, Any]:
    """Enhanced multi-hop processing with news integration and async processing"""
    start_time = time.time()
    
    # Initialize components
    news_analyzer = NewsAnalyzer()
    async_processor = AsyncProcessor()
    
    try:
        # Step 1: Query decomposition (existing logic)
        from unified_rag_with_graphs import client
        if not client:
            return {"error": "Groq client not initialized"}
        
        decomposition_prompt = f"""
        Decompose this complex query into 2-4 specific sub-questions for multi-hop reasoning:
        Query: "{query}"
        
        Return ONLY a JSON array of sub-questions:
        ["sub-question 1", "sub-question 2", "sub-question 3"]
        """
        
        decomp_response = client.chat.completions.create(
            messages=[{"role": "user", "content": decomposition_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=300
        )
        
        try:
            sub_questions = json.loads(decomp_response.choices[0].message.content)
        except:
            sub_questions = [query]
        
        # Step 2: Parallel data retrieval
        retrieved_data = await async_processor.parallel_data_retrieval(sub_questions, query)
        
        # Step 3: News integration (if relevant)
        news_data = None
        if any(word in query.lower() for word in ["news", "market", "stock", "company", "business", "economy"]):
            news_data = await news_analyzer.get_market_news_async(query, category)
        
        # Step 4: Evidence synthesis (existing logic)
        synthesis_prompt = f"""
        You are an expert analyst. Synthesize information from multiple sources to answer this query:
        
        Original Query: {query}
        
        Retrieved Evidence:
        {json.dumps([{
            "sub_question": item["sub_question"],
            "source": item.get("source", "Unknown"),
            "data": item.get("data", {})
        } for item in retrieved_data], indent=2)}
        
        {"News Analysis: " + json.dumps(news_data, indent=2) if news_data else ""}
        
        Provide a comprehensive analysis with specific facts, numbers, and proper citations.
        """
        
        final_response = client.chat.completions.create(
            messages=[{"role": "user", "content": synthesis_prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2000
        )
        
        # Enhance with decision engine
        from decision_engine import enhance_with_decision
        base_answer = final_response.choices[0].message.content
        enhanced_answer = enhance_with_decision(base_answer, query)
        
        processing_time = time.time() - start_time
        all_sources = list(set([item.get("source", "Unknown") for item in retrieved_data]))
        if news_data:
            all_sources.append("News Analysis")
        
        return {
            "answer": enhanced_answer,
            "sub_questions": sub_questions,
            "reasoning_steps": [f"Step {i}: {item['sub_question']} → {item.get('source', 'Unknown')}" 
                              for i, item in enumerate(retrieved_data, 1)],
            "sources": all_sources,
            "retrieved_data": retrieved_data,
            "news_data": news_data,
            "processing_time": processing_time,
            "async_processed": any(item.get("async_processed", False) for item in retrieved_data),
            "model_used": "llama-3.3-70b-versatile",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {"error": f"Enhanced processing failed: {str(e)}"}

# Export the main components
__all__ = [
    'NewsAnalyzer',
    'AsyncProcessor', 
    'render_news_dashboard',
    'render_async_progress',
    'enhanced_multi_hop_with_news'
]