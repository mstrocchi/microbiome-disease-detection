from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):

    @abstractmethod
    def encode(self, sequence: str) -> np.ndarray:
        pass
