import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec
import sqlite3
from study.getdata import preprocess_text


n = ['id', 'date', 'name', 'text', 'typr', 'rep', 'rtw', 'faw', 'stcount', 'foll', 'frien', 'listcount']
data_positive = pd.read_csv('../data/positive.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])
data_negative = pd.read_csv('../data/negative.csv', sep=';', error_bad_lines=False, names=n, usecols=['text'])

sample_size = min(data_positive.shape[0], data_negative.shape[0])
raw_data = np.concatenate((data_positive['text'].values[:sample_size],
                           data_negative['text'].values[:sample_size]), axis=0)
labels = [1] * sample_size + [0] * sample_size

data = [preprocess_text(t) for t in raw_data]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2)

conn = sqlite3.connect('../data/mysqlite3.db')
c = conn.cursor()

with open('../data/tweets.txt', 'w', encoding='utf-8') as f:
    for row in c.execute('SELECT ttext FROM sentiment'):
        if row[0]:
            tweet = preprocess_text(row[0])
            print(tweet, file=f)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
data = gensim.models.word2vec.LineSentence('../data/tweets.txt')
model = Word2Vec(data, size=200, window=5, min_count=3, workers=multiprocessing.cpu_count())

model.save("../models/w2v/model.w2v")