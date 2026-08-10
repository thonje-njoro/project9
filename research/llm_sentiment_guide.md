# LLM Sentiment Guide — Project9

## Provider: OpenRouter (Paid, ~$1/month)

**API Key:** `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`
**Model:** `meta-llama/llama-3.1-8b-instruct`
**Endpoint:** `https://openrouter.ai/api/v1/chat/completions`
**Cost:** ~$0.001/analysis (~$1/month for 1000 analyses)

## Test Results (8 headlines)

| Score | Headline |
|-------|----------|
| 0 | NVDA beats earnings expectations, stock surges 10 |
| -1 | Fed raises rates, markets tumble on recession fears |
| 0 | Apple announces new iPhone, stock stable |
| -1 | Tesla misses delivery targets, shares drop 5 |
| 0 | Microsoft cloud revenue exceeds forecasts, stock rallies |
| -1 | NFP comes in below forecast, unemployment rises |
| -1 | CPI inflation higher than expected, Fed hawkish |
| +1 | Strong jobs data boosts market confidence |

**Accuracy:** 5/8 correct (62.5%) — model is conservative on bullish signals

## Usage

```python
import sys, os
sys.path.insert(0, "/home/admin1/project9/backtest")
os.environ["OPENROUTER_API_KEY"] = "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"

from risk.llm_sentiment import LLMSentiment

sentiment = LLMSentiment(
    api_key=os.environ["OPENROUTER_API_KEY"],
    provider="openrouter"
)

score = sentiment.analyze_sentiment("NVDA beats earnings, stock surges")
# Returns: +1.0 (bullish), -1.0 (bearish), or 0.0 (neutral)
```

## Integration with Trading System

1. **Filter trades:** Only long when sentiment > 0, only short when sentiment < 0
2. **Adjust size:** Increase size when sentiment aligns with signal
3. **Skip on conflict:** Don't trade when sentiment conflicts with signal
