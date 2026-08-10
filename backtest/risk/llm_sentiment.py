"""
LLM Sentiment Analysis Module
==============================
Uses OpenRouter API (free tier) for sentiment analysis of financial news headlines.

Recommended Provider: OpenRouter (Paid, ~$0.001/analysis)
API: https://openrouter.ai/api/v1/chat/completions
Model: meta-llama/llama-3.1-8b-instruct
Cost: ~$1/month for 1000 analyses

Alternative Providers:
- NVIDIA NIM: https://integrate.api.nvidia.com/v1/chat/completions
- Groq: https://api.groq.com/openai/v1/chat/completions
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class LLMSentiment:
    """
    LLM-based sentiment analysis for financial news headlines.
    
    Usage:
        sentiment = LLMSentiment(api_key="YOUR_KEY", provider="openrouter")
        score = sentiment.analyze_sentiment("NVDA beats earnings expectations")
        # Returns: +1.0 (bullish), -1.0 (bearish), or 0.0 (neutral)
    """
    
    def __init__(
        self,
        api_key: str,
        provider: str = "openrouter",
        model: Optional[str] = None,
    ):
        """
        Initialize LLM Sentiment analyzer.
        
        Parameters
        ----------
        api_key : str
            API key for the provider
        provider : str
            Provider name: "openrouter", "nvidia", "groq"
        model : str, optional
            Model to use. If None, uses default for provider.
        """
        self.api_key = api_key
        self.provider = provider
        
        # Set endpoint and model based on provider
        if provider == "openrouter":
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model or "meta-llama/llama-3.1-8b-instruct"
        elif provider == "nvidia":
            self.endpoint = "https://integrate.api.nvidia.com/v1/chat/completions"
            self.model = model or "meta/llama-3.1-8b-instruct"
        elif provider == "groq":
            self.endpoint = "https://api.groq.com/openai/v1/chat/completions"
            self.model = model or "llama-3.1-8b-instant"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openrouter', 'nvidia', or 'groq'")
    
    def analyze_sentiment(self, headline: str) -> float:
        """
        Analyze sentiment of a financial news headline.
        
        Parameters
        ----------
        headline : str
            Financial news headline to analyze
        
        Returns
        -------
        float: Sentiment score
            +1.0 = BULLISH
             0.0 = NEUTRAL
            -1.0 = BEARISH
        """
        prompt = f"""Analyze the sentiment of this financial news headline for stock trading.

Headline: "{headline}"

Rate as:
- BULLISH (+1): Positive for stocks, good news, upgrades, beats expectations
- BEARISH (-1): Negative for stocks, bad news, downgrades, misses expectations
- NEUTRAL (0): No clear direction, informational only

Return ONLY the number (-1, 0, or +1). Nothing else."""

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,  # Low temperature for consistent results
                    "max_tokens": 10,
                },
                timeout=10,
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return 0.0
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            # Extract number from response
            if "+" in content or "1" in content:
                return 1.0
            elif "-" in content:
                return -1.0
            else:
                return 0.0
                
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return 0.0
    
    def analyze_batch(self, headlines: List[str]) -> List[float]:
        """
        Analyze sentiment for a batch of headlines.
        
        Parameters
        ----------
        headlines : List[str]
            List of headlines to analyze
        
        Returns
        -------
        List[float]: List of sentiment scores
        """
        scores = []
        for headline in headlines:
            score = self.analyze_sentiment(headline)
            scores.append(score)
        return scores
    
    def get_average_sentiment(self, headlines: List[str]) -> float:
        """
        Get average sentiment for a list of headlines.
        
        Parameters
        ----------
        headlines : List[str]
            List of headlines to analyze
        
        Returns
        -------
        float: Average sentiment score (-1.0 to +1.0)
        """
        if not headlines:
            return 0.0
        
        scores = self.analyze_batch(headlines)
        return sum(scores) / len(scores)


def get_news_headlines(
    symbol: str,
    api_key: str,
    hours_back: int = 24,
    max_headlines: int = 10,
) -> List[str]:
    """
    Get recent news headlines for a symbol using NewsAPI.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., "NVDA", "AAPL")
    api_key : str
        NewsAPI key (https://newsapi.org/)
    hours_back : int
        How many hours back to search (default: 24)
    max_headlines : int
        Maximum number of headlines to return (default: 10)
    
    Returns
    -------
    List[str]: List of news headlines
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': symbol,
        'from': (datetime.now() - timedelta(hours=hours_back)).isoformat(),
        'sortBy': 'publishedAt',
        'apiKey': api_key,
        'language': 'en',
        'pageSize': max_headlines,
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            return [article['title'] for article in articles if article.get('title')]
        else:
            print(f"NewsAPI Error: {response.status_code}")
            return []
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []


def calculate_sentiment_signal(
    symbol: str,
    llm_api_key: str,
    news_api_key: Optional[str] = None,
    provider: str = "openrouter",
) -> Dict:
    """
    Calculate sentiment signal for a symbol.
    
    Parameters
    ----------
    symbol : str
        Stock symbol
    llm_api_key : str
        LLM API key (OpenRouter, NVIDIA, or Groq)
    news_api_key : str, optional
        NewsAPI key for fetching headlines
    provider : str
        LLM provider to use
    
    Returns
    -------
    Dict with sentiment signal information
    """
    # Initialize sentiment analyzer
    sentiment = LLMSentiment(api_key=llm_api_key, provider=provider)
    
    # Get headlines if NewsAPI key provided
    if news_api_key:
        headlines = get_news_headlines(symbol, news_api_key)
    else:
        # Use sample headlines for testing
        headlines = [
            f"{symbol} reports strong quarterly earnings",
            f"{symbol} stock rises on positive analyst upgrade",
            f"{symbol} faces regulatory challenges",
        ]
    
    if not headlines:
        return {
            "symbol": symbol,
            "sentiment_score": 0.0,
            "headlines_count": 0,
            "signal": "NEUTRAL",
        }
    
    # Analyze sentiment
    scores = sentiment.analyze_batch(headlines)
    avg_score = sum(scores) / len(scores)
    
    # Determine signal
    if avg_score > 0.3:
        signal = "BULLISH"
    elif avg_score < -0.3:
        signal = "BEARISH"
    else:
        signal = "NEUTRAL"
    
    return {
        "symbol": symbol,
        "sentiment_score": avg_score,
        "headlines_count": len(headlines),
        "signal": signal,
        "individual_scores": scores,
        "headlines": headlines,
    }


def apply_sentiment_filter(
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    sentiment_scores: pd.Series,
    threshold: float = 0.3,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Apply sentiment filter to trading signals.
    
    Parameters
    ----------
    long_entries, long_exits, short_entries, short_exits : pd.Series (bool)
        Original trading signals
    sentiment_scores : pd.Series (float)
        Sentiment scores (-1.0 to +1.0)
    threshold : float
        Minimum sentiment to allow trades (default: 0.3)
    
    Returns
    -------
    Tuple of filtered signals
    """
    # Only allow long entries when sentiment is bullish
    filtered_long_entries = long_entries & (sentiment_scores > threshold)
    
    # Only allow short entries when sentiment is bearish
    filtered_short_entries = short_entries & (sentiment_scores < -threshold)
    
    # Exits remain unchanged
    filtered_long_exits = long_exits
    filtered_short_exits = short_exits
    
    return filtered_long_entries, filtered_long_exits, filtered_short_entries, filtered_short_exits


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("LLM SENTIMENT ANALYSIS — EXAMPLE")
    print("=" * 80)
    
    # Example with OpenRouter (free tier)
    # Replace with your actual API key
    API_KEY = "YOUR_OPENROUTER_API_KEY"
    
    print("\n1. Testing sentiment analysis...")
    
    # Test headlines
    test_headlines = [
        "NVDA beats earnings expectations, stock surges 10%",
        "Fed raises rates, markets tumble",
        "Apple announces new iPhone, stock stable",
        "Tesla misses delivery targets, shares drop",
        "Microsoft cloud revenue exceeds forecasts",
    ]
    
    # Note: This will fail without a valid API key
    # Uncomment and replace with your key to test
    
    # sentiment = LLMSentiment(api_key=API_KEY, provider="openrouter")
    # for headline in test_headlines:
    #     score = sentiment.analyze_sentiment(headline)
    #     print(f"  {headline[:50]}... -> {score:+.1f}")
    
    print("\n2. Example sentiment signal calculation...")
    
    # Example signal (without API call)
    example_signal = {
        "symbol": "NVDA",
        "sentiment_score": 0.7,
        "headlines_count": 5,
        "signal": "BULLISH",
    }
    
    print(f"  Symbol: {example_signal['symbol']}")
    print(f"  Sentiment: {example_signal['sentiment_score']:+.1f}")
    print(f"  Signal: {example_signal['signal']}")
    print(f"  Headlines: {example_signal['headlines_count']}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("1. Get API key from https://openrouter.ai/keys")
    print("2. Replace YOUR_OPENROUTER_API_KEY with your key")
    print("3. Test sentiment analysis")
    print("4. Integrate with trading system")
    print("=" * 80)
