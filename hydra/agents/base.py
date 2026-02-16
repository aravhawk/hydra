from abc import ABC, abstractmethod
from hydra.llm import LLMProvider
from hydra.types import Attack


class Agent(ABC):
    name: str
    role: str  # "red" or "blue"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @abstractmethod
    async def run(self, context: dict) -> list:
        """Execute the agent. Returns a list of Attacks or other results."""
        pass
