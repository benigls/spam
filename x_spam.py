#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.utils import np_utils

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve)

from spam.common import utils


np.random.seed(1337)

max_len = 800
max_words = 1000
batch_size = 64
classes = 2
epochs = 2
hidden_layers = [800, 500, 300, ]
noise_layers = [0.6, 0.4, ]

clean = lambda words: [str(word)
                       for word in words
                       if type(word) is not float]


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        try:
            loss = logs.get('loss')
        except AttributeError:
            loss = None

        self.losses.append(loss)


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

y_train = np.asarray(y_train, dtype='int32')
y_test = np.asarray(y_test, dtype='int32')

X_unlabel = tokenizer.texts_to_sequences(x_unlabel)
X_unlabel = pad_sequences(X_unlabel, maxlen=max_len, dtype='float64')

X_train = tokenizer.texts_to_sequences(x_train)
X_train = pad_sequences(X_train, maxlen=max_len, dtype='float64')

X_test = tokenizer.texts_to_sequences(x_test)
X_test = pad_sequences(X_test, maxlen=max_len, dtype='float64')

Y_train = np_utils.to_categorical(y_train, classes)
Y_test = np_utils.to_categorical(y_test, classes)

NPZ_DEST = 'data/npz'
print('Exporting npz files inside {}'.format(NPZ_DEST))
np.savez('{}/unlabel.npz'.format(NPZ_DEST), X=X_unlabel)
np.savez('{}/train.npz'.format(NPZ_DEST), X=X_train, y=y_train)
np.savez('{}/test.npz'.format(NPZ_DEST), X=X_test, y=y_test)

print('\n{}\n'.format('-' * 50))
print('Building model..')

encoders = []
input_data = np.copy(X_unlabel)

for i, (n_in, n_out) in enumerate(zip(
        hidden_layers[:-1], hidden_layers[1:]), start=1):
    print('Training layer {}: {} Layers -> {} Layers'
          .format(i, n_in, n_out))

    ae = Sequential()

    encoder = Dense(input_dim=n_in, output_dim=n_out,
                    activation='sigmoid')
    decoder = Dense(input_dim=n_out, output_dim=n_in,
                    activation='sigmoid')

    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False))
    ae.compile(loss='mean_squared_error', optimizer='rmsprop')

    pretraining_history = LossHistory()
    ae.fit(input_data, input_data, batch_size=batch_size,
           nb_epoch=epochs, callbacks=[pretraining_history])

    encoders.append(ae.layers[0].encoder)
    input_data = ae.predict(input_data)

model = Sequential()
for i, encoder in enumerate(encoders):
    model.add(GaussianNoise(noise_layers[i], input_shape=(hidden_layers[i],)))
    model.add(encoders[i])

model.add(Dense(input_dim=hidden_layers[-1], output_dim=classes,
                activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')

finetune_history = LossHistory()
model.fit(
    X_train, Y_train, batch_size=batch_size,
    nb_epoch=epochs, show_accuracy=True, callbacks=[finetune_history],
    validation_data=(X_test, Y_test), validation_split=0.1,
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(X_test)

metrics = {}
metrics['accuracy'] = accuracy_score(y_test, y_pred)
metrics['precision'] = precision_score(y_test, y_pred)
metrics['recall'] = recall_score(y_test, y_pred)
metrics['f1'] = f1_score(y_test, y_pred)

false_positive_rate, true_positive_rate, _ = \
    roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

for key, value in metrics.items():
    print('{}: {}'.format(key, value))

print('\n{}\n'.format('-' * 50))
print('Saving config results inside experiments/100_exp/')
exp_dir = 'experiments/exp_100'
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())

model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/metrics.json'.format(exp_dir), 'w') as f:
    json.dump(metrics, f, indent=4)

with open('{}/vocabulary.json'.format(exp_dir), 'w') as f:
    vocabulary = [w for w in tokenizer.word_counts]
    json.dump(vocabulary, f)

plt.figure(1)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = {}'.format(roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('{}/roc_curve.png'.format(exp_dir))

# TODO: add labels to loss history
plt.figure(2)
plt.title('Pretraining loss history')
plt.plot(pretraining_history.losses)
plt.savefig('{}/pretraining_loss.png'.format(exp_dir))

plt.figure(3)
plt.title('Finetune loss history')
plt.plot(finetune_history.losses)
plt.savefig('{}/finetune_loss.png'.format(exp_dir))

print('Done!')
