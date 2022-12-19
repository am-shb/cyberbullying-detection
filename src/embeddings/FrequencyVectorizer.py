# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 2022
@author: Amirmohammad Shahbandegan
"""
#import the necessary libraries
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from embeddings import BaseTextEmbeddingProvider


class FrequencyVectorizer(BaseTextEmbeddingProvider):
    def __init__(self, kind='tfidf', ngram_range=(1,1), max_df=1.0, min_df=1, max_features=None) -> None:
        super().__init__()
        if kind == 'bow':
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
        elif kind == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features)
        else:
            raise ValueError("kind must be either 'bow' or 'tfidf'")

    def fit(self, corpus: np.ndarray) -> None:
        self.vectorizer.fit(corpus)

    def save_pretrained(self, filename: str):
        with open(filename, 'wb+') as file:
            pickle.dump(self.vectorizer, file)

    def load_pretrained(self, filename: str):
        with open(filename, 'rb') as file:
            self.vectorizer = pickle.load(file)

    def encode(self, text: str) -> np.ndarray:
        return self.vectorizer.transform([text])[0]


    def batch_encode(self, text_arr: np.ndarray) -> np.ndarray:
        return self.vectorizer.transform(text_arr)
    

    def decode(self, encoded: np.ndarray) -> str:
        return self.vectorizer.inverse_transform(encoded)[0]


    def batch_decode(self, encoded_arr: np.ndarray) -> list:
        return self.vectorizer.inverse_transform(encoded_arr)
    
