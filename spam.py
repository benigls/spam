#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve,
                             confusion_matrix, matthews_corrcoef)

from spam.common import utils
from spam.dataset import EnronDataset
from spam.preprocess import Preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder, LossHistory


np.random.seed(1337)

CONFIG_FILENAME = 'config.json'

with open(CONFIG_FILENAME, 'r') as f:
    CONFIG = json.load(f)

if not CONFIG:
    print('Can\'t read config file.')
    sys.exit()

MODEL = CONFIG['model']
print('\n{}\n'.format('-' * 50))

if CONFIG['dataset']['generate']:
    print('Reading the dataset..')
    dataset = EnronDataset(path=CONFIG['dataset']['path'])

    enron_dataset = dataset.get_dataset()

    if CONFIG['dataset']['output']:
        print('Exporting the dataset..')
        dataset.to_csv(filepath=CONFIG['dataset']['filepath'])


print('Reading the dataset..')
if CONFIG['preprocess']['params']['read_csv']:
    preprocessor = Preprocess(**CONFIG['preprocess']['params'])
else:
    preprocessor = Preprocess(dataset=enron_dataset,
                              **CONFIG['preprocess']['params'])

if CONFIG['preprocess']['clean_dataset']:
    print('Cleaning the dataset..')
    preprocessor.clean_data()

if CONFIG['preprocess']['output_csv']:
    print('Exporting clean dataset..')
    preprocessor.dataset.to_csv(CONFIG['preprocess']['output_csv_filepath'])

print('Spliting the dataset..')
dataset = preprocessor.dataset

x_unlabel, (x_train, y_train), (x_test, y_test) = \
    utils.split_dataset(dataset['body'].values, dataset['label'].values)

print('Generating feature matrix and label vector..')
X_unlabel, X_train, X_test = preprocessor \
    .get_feature_matrix(x=(x_unlabel, x_train, x_test))

(y_train, Y_train), (y_test, Y_test) = preprocessor \
    .get_label_vector(y=(y_train, y_test))

vocabulary = preprocessor.vocabulary

import pdb
pdb.set_trace()

print('\n{}\n'.format('-' * 50))
print('Building model..')
sda = StackedDenoisingAutoEncoder(
    batch_size=CONFIG['model']['batch_size'],
    classes=CONFIG['model']['classes'],
    epochs=CONFIG['model']['epochs'],
    hidden_layers=CONFIG['model']['hidden_layers'],
    noise_layers=CONFIG['model']['noise_layers'],
    dataset=(X_unlabel, (X_train, y_train), (X_test, y_test)),
)
model, pretraining_history = sda.build_sda()

model.add(sda.build_finetune())

model.compile(loss=CONFIG['model']['finetune_loss'],
              optimizer=CONFIG['model']['finetune_optimizer'])

X_unlabel = sda.dataset['unlabel']
X_train, Y_train, y_train = sda.dataset['train']
X_test, Y_test, y_test = sda.dataset['test']

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')

finetune_history = LossHistory()
model.fit(
    X_train, Y_train, batch_size=sda.batch_size,
    nb_epoch=sda.epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
    callbacks=[finetune_history],
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

false_positive_rate, true_positive_rate, _ = \
    roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

for key, value in metrics.items():
    print('{}: {}'.format(key, value))

metrics['accuracy'] = accuracy_score(y_test, y_pred)
metrics['precision'] = precision_score(y_test, y_pred)
metrics['recall'] = recall_score(y_test, y_pred)
metrics['f1'] = f1_score(y_test, y_pred)
metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
metrics['auc'] = roc_auc

for key, value in metrics.items():
    print('{}: {}'.format(key, value))

print('\n{}\n'.format('-' * 50))
print('Saving config results inside experiments/100_exp/')
exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])
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

# print('Updating config id..')
# CONFIG['id'] += 1

# with open(CONFIG_FILENAME, 'w+') as f:
#     json.dump(CONFIG, f, indent=4)

print('Done!')
