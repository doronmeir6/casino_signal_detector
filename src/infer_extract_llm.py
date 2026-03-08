"""
Online LLM extractor for live inference.

Transforms a conversation into the canonical extraction schema shared with
training-time extraction.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from dotenv import load_dotenv

from src.signal_schema import validate_and_normalize_extraction

load_dotenv()

SYSTEM_PROMPT = """You are a casino guest-signal extractor.
Return strict JSON (no markdown) with this object shape:
{
  "guest_text":"...",
  "signals":{
    "intent":{"detected":bool,"confidence":0..1,"evidence":"..."},
    "value":{"detected":bool,"confidence":0..1,"evidence":"..."},
    "sentiment":{"detected":bool,"confidence":0..1,"evidence":"..."},
    "life_event":{"detected":bool,"confidence":0..1,"evidence":"..."},
    "competitive":{"detected":bool,"confidence":0..1,"evidence":"..."}
  },
  "parser_confidence":0..1,
  "notes":"..."
}
Use only evidence from guest turns.
"""


class LLMConversationExtractor:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_retries: int = 2,
    ):
        self.model = model
        self.max_retries = max_retries
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import openai

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is required for LLM extraction.")
            self._client = openai.OpenAI(api_key=api_key)
        return self._client

    def extract(self, conversation: list[dict], conversation_id: Optional[str] = None) -> dict:
        user_prompt = (
            "Extract signals from this conversation:\n"
            f"{json.dumps(conversation, ensure_ascii=False)}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        content = self._call(messages)
        if content is None:
            return self._fallback(conversation_id=conversation_id)

        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return self._fallback(conversation_id=conversation_id)

        ok, normalized, _ = validate_and_normalize_extraction(payload)
        normalized["conversation_id"] = conversation_id
        if not ok:
            return self._fallback(conversation_id=conversation_id)
        return normalized

    def _call(self, messages: list[dict]) -> Optional[str]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    max_tokens=700,
                )
                return resp.choices[0].message.content
            except Exception:
                if attempt == self.max_retries:
                    return None
                time.sleep(0.8 * attempt)
        return None

    @staticmethod
    def _fallback(conversation_id: Optional[str] = None) -> dict:
        # Safe fallback that never triggers positive predictions by itself.
        return {
            "conversation_id": conversation_id,
            "guest_text": "",
            "signals": {
                "intent": {"detected": False, "confidence": 0.0, "evidence": ""},
                "value": {"detected": False, "confidence": 0.0, "evidence": ""},
                "sentiment": {"detected": False, "confidence": 0.0, "evidence": ""},
                "life_event": {"detected": False, "confidence": 0.0, "evidence": ""},
                "competitive": {"detected": False, "confidence": 0.0, "evidence": ""},
            },
            "parser_confidence": 0.0,
            "notes": "fallback",
        }
