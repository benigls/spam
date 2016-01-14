#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve)

from spam.common import utils


np.random.seed(1337)

max_len = 800
max_words = 20000
batch_size = 64
classes = 2
epochs = 2
hidden_layers = [800, 600, 400, 300, ]
noise_layers = [0.6, 0.4, 0.3, ]

clean = lambda words: [str(word)
                       for word in words
                       if type(word) is not float]


print('\n{}\n'.format('-' * 50))
print('Reading csv files..')
dataset = pd.read_csv('data/csv/dataset.csv', encoding='iso-8859-1')

print('Spliting the dataset..')
x_unlabel, (x_train, y_train), (x_test, y_test) = \
    utils.split_dataset(dataset['body'].values,
                        dataset['label'].values)

print('Generating feature matrix..')
x_unlabel, x_train, x_test = \
    clean(x_unlabel), clean(x_train), clean(x_test)

tokenizer = Tokenizer(nb_words=max_words)
tokenizer.fit_on_texts(x_unlabel)

X_unlabel = tokenizer.texts_to_sequences(x_unlabel)
X_unlabel = pad_sequences(X_unlabel, maxlen=max_len, dtype='float64')

X_train = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(X_train, maxlen=max_len, dtype='float64')

X_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(X_test, maxlen=max_len, dtype='float64')

Y_train = np_utils.to_categorical(y_train, classes)
Y_true = np.asarray(y_test, dtype='int32')
Y_test = np_utils.to_categorical(y_test, classes)

print('\n{}\n'.format('-' * 50))
print('Building model..')

ae = Sequential()

encoder = containers.Sequential([
    GaussianNoise(0.5, input_shape=(800,)),
    Dense(input_dim=800, output_dim=400,
          activation='sigmoid', init='uniform')
])
decoder = Dense(input_dim=400, output_dim=800,
                activation='sigmoid')

ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                   output_reconstruction=False))
ae.compile(loss='mean_squared_error', optimizer='sgd')
ae.fit(X_unlabel, X_unlabel, batch_size=batch_size, nb_epoch=epochs)

model = Sequential()
model.add(ae.layers[0].encoder)
model.add(Dense(input_dim=400, output_dim=classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd')

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')
history = model.fit(
    X_train, Y_train, batch_size=batch_size,
    nb_epoch=epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

false_positive_rate, true_positive_rate, _ = \
    roc_curve(Y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1: {}'.format(f1))

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = {}'.format(roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('\n{}\n'.format('-' * 50))
print('Saving config results inside experiments/100_exp/')
exp_dir = 'experiments/exp_100'
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())
model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

plt.savefig('{}/roc_curve.png'.format(exp_dir))
print('Done!')
