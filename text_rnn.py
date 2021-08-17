'''
Emil Sjöberg

Text analysis using RNN
'''
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers import Flatten, Activation, Dense, Embedding, LSTM, GRU, SimpleRNN, Bidirectional
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams, make_sampling_table, pad_sequences
from keras.datasets import imdb

max_features = 10000  # number of words to consider as features
maxlen = 500  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

# Initializing model
lstm_big = Sequential()
lstm_big.add(Embedding(max_features, 128))
lstm_big.add(LSTM(128, return_sequences=True))
lstm_big.add(LSTM(128))
lstm_big.add(Dense(1, activation='sigmoid'))

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)


# Big LSTM model
opt = keras.optimizers.Adam(learning_rate=0.0001)
lstm_big.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
history = lstm_big.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stopping])

_, acc = lstm_big.evaluate(x_test, y_test)
print('Accuracy = ', acc)
