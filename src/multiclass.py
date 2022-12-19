# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 12:03:17 2020
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


base_path = '/scratch/ashahban/cb'

def base_dir(file_name):
  return os.path.join(base_path, file_name)

#load the packaged embeddings 
with open(base_dir('embeddings.pkl'), 'rb') as f:
  embeddings = pickle.load(f)

df = pd.read_csv(base_dir('cyberbullying_tweets.csv'))
ncb_idx = df[df['cyberbullying_type'] == 'not_cyberbullying'].index
labels = df.drop(ncb_idx)['cyberbullying_type'].to_numpy()

for e, emb in embeddings.items():
  embeddings[e] = emb[7945:]

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB


#Classifiers with default hyper parameters 
classifiers = {
    'lr': LogisticRegression(),
    'svm': SVC(),
    'lsvm': LinearSVC(),
#     'knn': KNeighborsClassifier(),
    'mlp': MLPClassifier(learning_rate='adaptive', max_iter=100),
    'xgb': xgb.XGBClassifier(),
}

#The model is fitted with the data and training and testing is performed using the above classifiers
def run_experiment(X_train, X_test, y_train, y_test, clf, e, c, f):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

#The evaluation metrics used are f1_score and accuracy
  f1_micro = f1_score(y_test, y_pred, average='micro')
  f1_macro = f1_score(y_test, y_pred, average='macro')
  f1_weighted = f1_score(y_test, y_pred, average='weighted')
  accuracy = accuracy_score(y_test, y_pred)

  l = []
  l.append('m')
  l.append(str(e))
  l.append(str(c))
  l.append(str(f))
  l.append(str(f1_micro))
  l.append(str(f1_macro))
  l.append(str(f1_weighted))
  l.append(str(accuracy))

  print(','.join(l))
  
  with open(base_dir('results/m_{}_{}_{}.txt'.format(e,c,f)), 'w+') as out:
    out.write(','.join(l) + '\n')

kf = KFold(n_splits=5, shuffle=True, random_state=0)

processes = []
for c, clf in classifiers.items():
  for e, emb in embeddings.items():
    for f, (train_index, test_index) in enumerate(kf.split(emb)):
      if os.path.exists(base_dir('results/m_{}_{}_{}.txt'.format(e,c,f))):
        continue

      X_train, X_test = emb[train_index], emb[test_index]
      y_train, y_test = labels[train_index], labels[test_index]
      
      p = multiprocessing.Process(
          target=run_experiment,
          args = [
                  X_train,
                  X_test,
                  y_train,
                  y_test,
                  clf,
                  e,
                  c,
                  f
          ])
      p.start()
      processes.append(p) 
for p in processes:
  p.join()
