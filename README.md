# Cyberbullying Detection Project - (2022W) COMP-5012-WDE - Big Data

## Abstract
Cyberbullying can have serious legal consequences in Canada including jail time and fines. Social media companies such as Twitter, Facebook etc., have resources and guides on cyberbullying and are relying on passive reporting mechanisms. However, 90\% of cyberbullying activities go unreported making the presence of an active cyberbullying detection system crucial. Our proposed architecture detects cyberbullying using a two-step multiclass classification method using traditional machine learning algorithms in a balanced dataset distributed into six cyberbullying classes. Our model tackles both balanced classes and imbalanced classes in the dataset and outperforms the current ML and DNN baselines. This work experiments with multiple text embedding methods to compare and find the most suitable strategy in detecting cyberbullying. Our results provide significant insights into the effectiveness of constructing architectures using traditional ML models rather than implementing deep learning methods to overcome the cyberbullying issue. We have released our models and code.

## Implemetation
The source code is implemented using python 3.8. There are 3 main directories in this repo: `data`, `src`, `jobs`. The `data` directory holds the dataset and the `jobs` directory contains the scripts used to submit the jobs to *Compute Canada* nodes. All of the source code resides in the `src` directory.

The embedding generators are defined using an interface defined in `BaseTextEmbeddingProvider` class. This interface standardizes the methods used to generate text embeddings in future calls. 6 different embedding methods are implemented in 3 classes called `FrequencyVectorizer`, `Word2vecVectorizer`, and `SbertVectorizer`.
The embeddings are then generated and saved into a file using the notebook file `embeddings.ipynb`. The stored embeddings are used in the further steps.

The rest of the python scripts take care of the classification models. These models are written in a way that they can take advantage of a multi-core system. `multiclass.py` and `multiclass6.py` are responsible for the multi-class classification models. `binary.py`, `rbinary.py`, and `nbinary.py` are implementing binary models without under sampling, with random under sampling, and with near miss under sampling respectively. 
The `final.py` file houses the final pipeline model with the best classifiers.

The required packages to run the code are listed below.
```
numpy
scipy
pandas
imblearn
scikit-learn
gensim
emoji
nltk
spacy
contractions
sentence_transformers
```

## Authors
- Amirmohammad Shahbandegan
- Lakshmi Preethi Kamak
- Mohammad Ghadiri
