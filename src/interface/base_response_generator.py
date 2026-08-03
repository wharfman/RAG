from abc import ABC, abstractmethod
from typing import List

class BaseResponeseGenerator(ABC):

    @abstractmethod
    def generate_response(self, query: str, context: List[str]) -> str:
        pass