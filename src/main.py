import pandas as pd
import pickle

from embeddings import FrequencyVectorizer, Word2vecVectorizer, SbertVectorizer

df = pd.read_csv('../data/cyberbullying_tweets.csv')

tweets = df['tweet_text'].to_numpy()
labels = df['cyberbullying_type'].to_numpy()
binary_labels = ~(df['cyberbullying_type'] == 'not_cyberbullying').to_numpy()

embedders = {
    'bow': FrequencyVectorizer('bow', ngram_range=(1,3), max_features=30000),
    'tfidf': FrequencyVectorizer('tfidf', ngram_range=(1,3), max_features=30000),
    'w2v': Word2vecVectorizer('word2vec-google-news-300'),
    'golve': Word2vecVectorizer('glove-wiki-gigaword-300'),
    'ft': Word2vecVectorizer('fasttext-wiki-news-subwords-300'),
    'sbert': SbertVectorizer('all-mpnet-base-v2')
}

embedders['bow'].fit(tweets)
embedders['tfidf'].fit(tweets)

embeddings = {}
for name, embedder in embedders.items():
    print(name)
    embeddings[name] = embedder.batch_encode(tweets)

with open('embeddings.pkl', 'wb+') as file:
    pickle.dump(embeddings, file)