from openai import AsyncOpenAI


class Target:
    """An AI system being tested."""

    def __init__(self, name: str, base_url: str, model: str,
                 system_prompt: str = "You are a helpful assistant."):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key="ollama",
        )

    async def send(self, prompt: str) -> str:
        """Send a prompt to the target and return its response."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            return f"[TARGET ERROR: {e}]"
