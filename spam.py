#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve,
                             confusion_matrix, matthews_corrcoef)

from spam.common import utils
from spam.preprocess import preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder


def parse_config():
    """ Parses the args and return the config file in json. """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', action='store')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            config = json.load(f)

        return config, args.config
    except Exception as e:
        print('Error: {}'.format(e))
        return None, None


np.random.seed(1337)

CONFIG, CONFIG_FILENAME = parse_config()
if not CONFIG:
    sys.exit()

NPZ_DEST = CONFIG['npz']['dest']
CSV_DEST = CONFIG['csv']['dest']

if CONFIG['csv']['generate']:
    file_path_list = utils.get_file_path_list(
        utils.dataset_meta(CONFIG['dataset']))

    # transform list of tuple into two list
    # e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
    paths, labels = zip(*file_path_list)

    # generate panda dataframes and export it to csv
    print('\n{}\n'.format('-' * 50))
    print('Generating dataframe..')
    dataset = pd.DataFrame(**utils.df_params(paths, labels))

    print('\nExporting dataframe into a csv file inside {}'
          .format(CSV_DEST))
    dataset.to_csv('{}/dataset.csv'.format(CSV_DEST))

if CONFIG['npz']['generate']:
    print('\n{}\n'.format('-' * 50))
    print('Reading csv files..')
    dataset = pd.read_csv('{}/dataset.csv'
                          .format(CSV_DEST),
                          encoding='iso-8859-1')

    print('Spliting the dataset..')
    x_unlabel, (x_train, y_train), (x_test, y_test) = \
        utils.split_dataset(dataset['body'].values,
                            dataset['label'].values)

    y_train = np.assarray(y_train, dtype='int32')
    y_test = np.assarray(y_test, dtype='int32')

    print('Generating feature matrix..')
    X_unlabel, X_train, X_test = preprocess.feature_matrix(
        dataset=[x_unlabel, x_train, x_test, ],
        max_words=CONFIG['preprocess']['max_words'],
        max_len=CONFIG['preprocess']['max_len']
    )

    print('Exporting npz files inside {}'.format(NPZ_DEST))
    np.savez('{}/unlabel.npz'.format(NPZ_DEST), X=X_unlabel)
    np.savez('{}/train.npz'.format(NPZ_DEST), X=X_train, y=y_train)
    np.savez('{}/test.npz'.format(NPZ_DEST), X=X_test, y=y_test)

print('\n{}\n'.format('-' * 50))
print('Building model..')
sda = StackedDenoisingAutoEncoder(
    batch_size=CONFIG['model']['batch_size'],
    classes=CONFIG['model']['classes'],
    epochs=CONFIG['model']['epochs'],
    hidden_layers=CONFIG['model']['hidden_layers'],
    noise_layers=CONFIG['model']['noise_layers'],
)
model = sda.build_sda()

model.add(sda.build_finetune())

model.compile(loss='categorical_crossentropy', optimizer='sgd')

X_train, Y_train = sda.dataset['train']
X_test, Y_test, Y_true = sda.dataset['test']

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')
history = model.fit(
    X_train, Y_train, batch_size=sda.batch_size,
    nb_epoch=sda.epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(X_test)

accuracy = accuracy_score(Y_true, y_pred)
precision = precision_score(Y_true, y_pred)
recall = recall_score(Y_true, y_pred)
f1 = f1_score(Y_true, y_pred)

false_positive_rate, true_positive_rate, _ = \
    roc_curve(Y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print(y_pred)
print(Y_true)

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
print('Saving config results inside experiments/{}_exp/ ..'
      .format(CONFIG['id']))
exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())
model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/config.json'.format(exp_dir), 'w') as f:
    json.dump(CONFIG, f, indent=4)

plt.savefig('{}/roc_curve.png'.format(exp_dir))

# print('Updating config id..')
# CONFIG['id'] += 1

# with open(CONFIG_FILENAME, 'w+') as f:
#     json.dump(CONFIG, f, indent=4)

print('Done!')
