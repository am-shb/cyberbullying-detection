# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 2022
@author: Amirmohammad Shahbandegan
"""
#import the necessary libraries
from typing import Iterable
import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import BaseTextEmbeddingProvider

models = ['all-mpnet-base-v2']

class SbertVectorizer(BaseTextEmbeddingProvider):
    def __init__(self, model) -> None:
        super().__init__()
        if model not in models:
            raise ValueError("model must be one of the following: " + ', '.join(models))

        self.model_name = model
        self.model = SentenceTransformer(model)


    def encode(self, text: str) -> np.ndarray:
        return self.model.encode(text)


    def batch_encode(self, text_arr: Iterable) -> np.ndarray:
        return self.model.encode(text_arr)
    
    
