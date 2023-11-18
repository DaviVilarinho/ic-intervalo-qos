from abc import ABC, abstractmethod
import pandas as pd


class SparseStrategy(ABC):
    @abstractmethod
    def sparse(self, x: pd.DataFrame, y: pd.DataFrame, sparsing_factor: int):
        pass

class NaiveSparseStrategy(SparseStrategy):
    def sparse(self, x, y, sparsing_factor):
        x = x[x['Timestamp'] % sparsing_factor == 0]
        y = y[y['Timestamp'] % sparsing_factor == 0]
        return x, y
