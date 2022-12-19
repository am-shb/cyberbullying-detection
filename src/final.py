# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 2022
@author: Amirmohammad Shahbandegan
"""
#import the necessary libraries

import numpy as np
import pandas as pd
import multiprocessing
import os
import pickle

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB

from imblearn.under_sampling import NearMiss, RandomUnderSampler


base_path = '/scratch/ashahban/cb'


#load the packaged embeddings and labels
def base_dir(file_name):
  return os.path.join(base_path, file_name)

with open(base_dir('embeddings.pkl'), 'rb') as f:
  embeddings = pickle.load(f)

df = pd.read_csv(base_dir('cyberbullying_tweets.csv'))
labels = df['cyberbullying_type'].to_numpy()
binary_labels = ~(df['cyberbullying_type'] == 'not_cyberbullying').to_numpy()

#KFold Validation
kf = KFold(n_splits=5, shuffle=True, random_state=0)

#Two stage pipeline model
for f, (train_index, test_index) in enumerate(kf.split(labels)):
  b_X_train, b_X_test = embeddings['sbert'][train_index], embeddings['sbert'][test_index]
  b_y_train, b_y_test = binary_labels[train_index], binary_labels[test_index]

  m_X_train, m_X_test = embeddings['bow'][train_index[b_y_train]], embeddings['bow'][test_index]
  m_y_train, y_test = labels[train_index[b_y_train]], labels[test_index]
  
  binary_clf = SVC()
  binary_clf.fit(b_X_train, b_y_train)
  m_clf = xgb.XGBClassifier()
  m_clf.fit(m_X_train, m_y_train)
  
  y_pred = np.zeros_like(y_test)
  b_pred = binary_clf.predict(b_X_test)
  m_pred = m_clf.predict(m_X_test[b_pred])
  y_pred[b_pred] = m_pred
  y_pred[~b_pred] = 'not_cyberbullying'
#The evaluation metrics used are f1_score and accuracy
  f1_micro = f1_score(y_test, y_pred, average='micro')
  f1_macro = f1_score(y_test, y_pred, average='macro')
  f1_weighted = f1_score(y_test, y_pred, average='weighted')
  accuracy = accuracy_score(y_test, y_pred)

  l = []
  l.append('f')
  l.append(str(f))
  l.append(str(f1_micro))
  l.append(str(f1_macro))
  l.append(str(f1_weighted))
  l.append(str(accuracy))

  with open(base_dir('resultsf/f_{}.txt'.format(f)), 'w+') as out:
    out.write(','.join(l) + '\n')
