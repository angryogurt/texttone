import numpy as np
from gensim.models import Word2Vec
from keras.layers import Input
from keras.layers.embeddings import Embedding
from keras.layers import Dense, concatenate, Activation, Dropout
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras import optimizers
import pickle

with open('../models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

w2v_model = Word2Vec.load('models/w2v/model.w2v')
DIM = w2v_model.vector_size
embedding_matrix = np.zeros((100000, DIM))
for word, i in tokenizer.word_index.items():
    if i >= 100000:
        break
    if word in w2v_model.wv.vocab.keys():
        embedding_matrix[i] = w2v_model.wv[word]

tweet_input = Input(shape=(26,), dtype='int32')
tweet_encoder = Embedding(100000, DIM, input_length=26,
                          weights=[embedding_matrix], trainable=False)(tweet_input)

branches = []
x = Dropout(0.2)(tweet_encoder)
for size, filters_count in [(2, 10), (3, 10), (4, 10), (5, 10)]:
    for i in range(filters_count):
        branch = Conv1D(filters=1, kernel_size=size, padding='valid', activation='relu')(x)
        branch = GlobalMaxPooling1D()(branch)
        branches.append(branch)
x = concatenate(branches, axis=1)
x = Dropout(0.2)(x)
x = Dense(30, activation='relu')(x)
x = Dense(1)(x)
output = Activation('sigmoid')(x)
model = Model(inputs=[tweet_input], outputs=[output])
model.load_weights('../models/cnn/cnn-trainable.hdf5')
model.layers[1].trainable = True
adam = optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=adam)

model.save('../models/cnn/CNN.h5')