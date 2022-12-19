# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 2022
@author: Amirmohammad Shahbandegan
"""
#import the necessary libraries
from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np


class BaseTextEmbeddingProvider(ABC):
    def __init__(self) -> None:
        pass

    def fit(self, corpus: np.ndarray) -> None:
        raise NotImplemented()

    def load_pretrained(self) -> None:
        raise NotImplemented()

    def save_pretrained(self) -> None:
        raise NotImplemented()

    @abstractmethod
    def encode(self, text: str) -> np.ndarray:
        pass

    @abstractmethod
    def batch_encode(self, text_arr: Iterable) -> np.ndarray:
        pass

    def decode(self, encoded: np.ndarray) -> str:
        raise NotImplemented()

    def batch_decode(self, encoded_arr: np.ndarray) -> np.ndarray:
        raise NotImplemented()
