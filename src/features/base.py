from abc import ABC, abstractmethod
import pandas as pd


class BaseFeatureStrategy(ABC):

    @abstractmethod
    def construct(self, df: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
        pass