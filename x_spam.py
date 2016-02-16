#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback
from keras.utils import np_utils

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve,
                             confusion_matrix)

from spam.common import utils


start_time = timeit.default_timer()

np.random.seed(1337)

exp_num = 100
max_len = 800
max_words = 1000
batch_size = 256
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
noises = []
input_data = np.copy(X_unlabel)

for i, (n_in, n_out) in enumerate(zip(
        hidden_layers[:-1], hidden_layers[1:]), start=1):
    print('Training layer {}: {} Layers -> {} Layers'
          .format(i, n_in, n_out))

    ae = Sequential()

    # encoder = Dense(input_dim=n_in, output_dim=n_out,
    #                 activation='sigmoid')
    encoder = containers.Sequential([
        GaussianNoise(noise_layers[i - 1], input_shape=(n_in,)),
        Dense(input_dim=n_in, output_dim=n_out,
              activation='sigmoid', init='uniform'),
    ])
    decoder = Dense(input_dim=n_out, output_dim=n_in,
                    activation='sigmoid')

    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False))
    ae.compile(loss='mean_squared_error', optimizer='adadelta')

    pretraining_history = LossHistory()
    ae.fit(input_data, input_data, batch_size=batch_size,
           nb_epoch=epochs, callbacks=[pretraining_history])

    encoders.append(ae.layers[0].encoder.layers[1])
    noises.append(ae.layers[0].encoder.layers[0])
    input_data = ae.predict(input_data)

model = Sequential()
for encoder, noise in zip(encoders, noises):
    # model.add(GaussianNoise(noise_layers[i],
    #                         input_shape=(hidden_layers[i],)))
    model.add(noise)
    model.add(encoder)

model.add(Dense(input_dim=hidden_layers[-1], output_dim=classes,
                activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

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
data_meta = {}

data_meta['unlabeled_count'] = len(X_unlabel)
data_meta['labeled_count'] = len(X_train) + len(X_test)
data_meta['train_data'] = {}
data_meta['test_data'] = {}

data_meta['train_data']['spam_count'] = int(sum(y_train))
data_meta['train_data']['ham_count'] = int(len(y_train) - sum(y_train))
data_meta['train_data']['total_count'] = \
    data_meta['train_data']['spam_count'] + \
    data_meta['train_data']['ham_count']

data_meta['test_data']['spam_count'] = int(sum(y_test))
data_meta['test_data']['ham_count'] = int(len(y_test) - sum(y_test))
data_meta['test_data']['total_count'] = \
    data_meta['test_data']['spam_count'] + \
    data_meta['test_data']['ham_count']

conf_matrix = confusion_matrix(y_test, y_pred)

metrics['true_positive'], metrics['true_negative'], \
    metrics['false_positive'], metrics['false_negative'] = \
    int(conf_matrix[0][0]), int(conf_matrix[1][1]), \
    int(conf_matrix[0][1]), int(conf_matrix[1][0])

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
exp_dir = 'experiments/exp_{}'.format(exp_num)
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())

model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/metrics.json'.format(exp_dir), 'w') as f:
    json.dump(metrics, f, indent=4)

with open('{}/data_meta.json'.format(exp_dir), 'w') as f:
    json.dump(data_meta, f, indent=4)

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

end_time = timeit.default_timer()

print('Done!')
print('Run for %.2fm' % ((end_time - start_time) / 60.0))
