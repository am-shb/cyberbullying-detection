# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 2022
@author: Amirmohammad Shahbandegan
"""
#import the necessary libraries
import pickle
from typing import Iterable
import numpy as np
import gensim.downloader
from gensim.utils import tokenize

from embeddings import BaseTextEmbeddingProvider

models = ['word2vec-google-news-300', 'glove-wiki-gigaword-300', 'fasttext-wiki-news-subwords-300']

class Word2vecVectorizer(BaseTextEmbeddingProvider):
    def __init__(self, model) -> None:
        super().__init__()
        if model not in models:
            raise ValueError("model must be one of the following: " + ', '.join(models))

        self.model_name = model
        self.model = gensim.downloader.load(model)


    def encode(self, text: str) -> np.ndarray:
        encoded = np.array([self.model[token] for token in tokenize(text) if token in self.model]).mean(axis=0)
        if encoded.shape != (300,):
            return np.zeros((300,))
        return encoded


    def batch_encode(self, text_arr: Iterable) -> np.ndarray:
        return np.array([self.encode(s) for s in text_arr])
    
    
