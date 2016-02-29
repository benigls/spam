#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import timeit

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve,
                             confusion_matrix, matthews_corrcoef)

from spam.common import utils
from spam.deeplearning import LossHistory
from spam.preprocess import Preprocess


start_time = timeit.default_timer()

np.random.seed(1337)

exp_num = 100

preprocess_params = {
    'max_words': 1000,
    'max_len': 800,
    'mode': 'tfidf',
    'read_csv': True,
    'read_csv_filepath': 'data/csv/clean_dataset.csv',
    'classes': 2,
}

epochs = 50
batch_size = 128
classes = 2
hidden_layers = [800, 500, 300, ]
noise_layers = [0.6, 0.4, ]
pretr_activ = 'sigmoid'
pretr_opt = 'adadelta'
pretr_loss = 'mse'
fine_activ = 'softmax'
fine_opt = 'adadelta'
fine_loss = 'categorical_crossentropy'

clean = lambda words: [str(word)
                       for word in words
                       if type(word) is not float]

print('\n{}\n'.format('-' * 50))
print('Reading the dataset..')
preprocessor = Preprocess(**preprocess_params)

print('Spliting the dataset..')
enron_dataset = preprocessor.dataset
enron_dataset = utils.split_dataset(x=enron_dataset['body'].values,
                                    y=enron_dataset['label'].values)

print('Transforming dataset into vectors and matrices..')
enron_dataset = preprocessor.transform(dataset=enron_dataset)
vocabulary = preprocessor.vocabulary

print('\n{}\n'.format('-' * 50))
print('Building model..')

encoders = []
noises = []
pretraining_history = []

input_data = np.copy(enron_dataset.unlabel)

print('Pretraining model..')
for i, (n_in, n_out) in enumerate(zip(
        hidden_layers[:-1], hidden_layers[1:]), start=1):
    print('Training layer {}: {} Layers -> {} Layers'
          .format(i, n_in, n_out))

    ae = Sequential()

    encoder = containers.Sequential([
        GaussianNoise(noise_layers[i - 1], input_shape=(n_in,)),
        Dense(input_dim=n_in, output_dim=n_out,
              activation=pretr_activ, init='uniform'),
    ])
    decoder = Dense(input_dim=n_out, output_dim=n_in,
                    activation=pretr_activ)

    ae.add(AutoEncoder(encoder=encoder, decoder=decoder,
                       output_reconstruction=False))
    ae.compile(loss=pretr_loss, optimizer=pretr_opt)

    temp_history = LossHistory()
    ae.fit(input_data, input_data, batch_size=batch_size,
           nb_epoch=epochs, callbacks=[temp_history])

    pretraining_history += temp_history.losses
    encoders.append(ae.layers[0].encoder.layers[1])
    noises.append(ae.layers[0].encoder.layers[0])
    input_data = ae.predict(input_data)

model = Sequential()
for encoder, noise in zip(encoders, noises):
    model.add(noise)
    model.add(encoder)

model.add(Dense(input_dim=hidden_layers[-1], output_dim=classes,
                activation=fine_activ))

model.compile(loss=fine_loss, optimizer=fine_opt)

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')

finetune_history = LossHistory()
model.fit(
    enron_dataset.train.X, enron_dataset.train.Y,
    batch_size=batch_size,
    nb_epoch=epochs, show_accuracy=True,
    validation_data=(enron_dataset.test.X, enron_dataset.test.Y),
    validation_split=0.1,
    callbacks=[finetune_history],
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(enron_dataset.test.X)

metrics = {}
data_meta = {}

data_meta['unlabeled_count'] = len(enron_dataset.unlabel)
data_meta['labeled_count'] = \
    len(enron_dataset.train.X) + len(enron_dataset.test.X)

data_meta['train_data'] = {}
data_meta['test_data'] = {}

data_meta['train_data']['spam_count'] = int(sum(enron_dataset.train.y))
data_meta['train_data']['ham_count'] = \
    int(len(enron_dataset.train.y) - sum(enron_dataset.train.y))
data_meta['train_data']['total_count'] = \
    data_meta['train_data']['spam_count'] + \
    data_meta['train_data']['ham_count']

data_meta['test_data']['spam_count'] = int(sum(enron_dataset.test.y))
data_meta['test_data']['ham_count'] = \
    int(len(enron_dataset.test.y) - sum(enron_dataset.test.y))
data_meta['test_data']['total_count'] = \
    data_meta['test_data']['spam_count'] + \
    data_meta['test_data']['ham_count']

conf_matrix = confusion_matrix(enron_dataset.test.y, y_pred)

metrics['true_positive'], metrics['true_negative'], \
    metrics['false_positive'], metrics['false_negative'] = \
    int(conf_matrix[0][0]), int(conf_matrix[1][1]), \
    int(conf_matrix[0][1]), int(conf_matrix[1][0])

false_positive_rate, true_positive_rate, _ = \
    roc_curve(enron_dataset.test.y, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

metrics['accuracy'] = accuracy_score(enron_dataset.test.y, y_pred)
metrics['precision'] = precision_score(enron_dataset.test.y, y_pred)
metrics['recall'] = recall_score(enron_dataset.test.y, y_pred)
metrics['f1'] = f1_score(enron_dataset.test.y, y_pred)
metrics['mcc'] = matthews_corrcoef(enron_dataset.test.y, y_pred)
metrics['auc'] = roc_auc

for key, value in metrics.items():
    print('{}: {}'.format(key, value))

print('\n{}\n'.format('-' * 50))

exp_dir = 'experiments/exp_{}'.format(exp_num)
print('Saving config results inside {}'.format(exp_dir))

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
plt.plot(pretraining_history)
plt.savefig('{}/pretraining_loss.png'.format(exp_dir))

plt.figure(3)
plt.title('Finetune loss history')
plt.plot(finetune_history.losses)
plt.savefig('{}/finetune_loss.png'.format(exp_dir))

end_time = timeit.default_timer()

print('Done!')
print('Run for %.2fm' % ((end_time - start_time) / 60.0))
