"""LLM-based signal validator for paper trading.

Uses a lightweight LLM call to validate trading signals before execution.
Inspired by TradingAgents' risk management team.
"""

import json
import os
from typing import Optional

import pandas as pd
import numpy as np


class SignalValidator:
    """Validates trading signals using LLM reasoning."""

    def __init__(
        self,
        provider: str = "mimo",
        model: str = "mimo-v2.5",
        max_tokens: int = 200,
        temperature: float = 0.3,
        min_confidence: float = 0.6,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_confidence = min_confidence

    def validate(
        self,
        symbol: str,
        action: str,
        price: float,
        regime: str,
        context: str,
    ) -> dict:
        """Validate a trading signal using LLM.

        Args:
            symbol: Ticker symbol.
            action: Proposed action.
            price: Current price.
            regime: Current regime.
            context: Pre-built context string.

        Returns:
            dict with keys: approved (bool), confidence (float), reasoning (str)
        """
        prompt = f"""You are a risk manager validating a trading signal.

{context}

Should this {action} signal be approved? Consider:
1. Is the price action consistent with the proposed direction?
2. Is the volatility regime appropriate for this trade?
3. Are there any warning signs (overbought/oversold, extreme moves)?

Respond with ONLY a JSON object:
{{"approved": true/false, "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            result = self._call_llm(prompt)
            return self._parse_response(result)
        except Exception as e:
            return {
                "approved": True,
                "confidence": 0.5,
                "reasoning": f"LLM validation failed: {e}. Defaulting to approve.",
            }

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM provider."""
        if self.provider == "mimo":
            return self._call_mimo(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_mimo(self, prompt: str) -> str:
        """Call MiMo via the local inference endpoint."""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception:
            return '{"approved": true, "confidence": 0.5, "reasoning": "MiMo unavailable"}'

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        try:
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception:
            return '{"approved": true, "confidence": 0.5, "reasoning": "OpenAI unavailable"}'

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception:
            return '{"approved": true, "confidence": 0.5, "reasoning": "Anthropic unavailable"}'

    def _parse_response(self, response: str) -> dict:
        """Parse LLM response into structured output."""
        try:
            text = response.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
            result = json.loads(text)
            return {
                "approved": bool(result.get("approved", True)),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception:
            return {
                "approved": True,
                "confidence": 0.5,
                "reasoning": f"Failed to parse response: {response[:100]}",
            }
