import asyncio
import json
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMProvider:
    def __init__(self, base_url: str, model: str):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",  # Ollama doesn't require a real key
        )
        self.model = model

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """Send a chat request and return the response text."""
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise

    async def chat_json(self, messages: list[dict], temperature: float = 0.7) -> dict | list:
        """Send a chat request and parse the response as JSON."""
        raw = await self.chat(messages, temperature)
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # Remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)
        return json.loads(cleaned)
