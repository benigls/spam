#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from keras.optimizers import SGD
from spam.common import DATASET_META
from spam.common.utils import get_file_path_list, split_dataset
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

        return config
    except Exception as e:
        print('Error: {}'.format(e))
        return None


CONFIG = parse_config()
if not CONFIG:
    sys.exit()

UNLABEL = -1
HAM = 0
SPAM = 1
NPZ_DEST = CONFIG['npz']['dest']
CSV_DEST = CONFIG['csv']['dest']

if CONFIG['csv']['generate']:
    file_path_list = get_file_path_list(DATASET_META)

    print('Spliting the dataset..')
    unlabeled_path, (train_path, train_class), \
        (test_path, test_class) = split_dataset(file_path_list, seed=1337)

    # generate panda dataframes and export it to csv
    print('Generating unlabeled dataframe..')
    unlabeled_data = pd.DataFrame(
        data={
            'email': [preprocess.read_email(path) for path in unlabeled_path],
            'class': [UNLABEL for _ in range(len(unlabeled_path))]
        },
        columns=['email', 'class'],
    )

    print('\nGenerating train dataframe..')
    train_data = pd.DataFrame(
        data={
            'email': [preprocess.read_email(path) for path in train_path],
            'class': [SPAM if cl == 'spam' else HAM for cl in train_class]
        },
        columns=['email', 'class'],
    )

    print('\nGenerating test dataframe..')
    test_data = pd.DataFrame(
        data={
            'email': [preprocess.read_email(path) for path in test_path],
            'class': [SPAM if cl == 'spam' else HAM for cl in test_class]
        },
        columns=['email', 'class'],
    )

    print('\nExporting dataframes into a csv files inside {} ..'
          .format(CSV_DEST))
    unlabeled_data.to_csv('{}/unlabeled_data.csv'.format(CSV_DEST))
    train_data.to_csv('{}/train_data.csv'.format(CSV_DEST))
    test_data.to_csv('{}/test_data.csv'.format(CSV_DEST))

if CONFIG['npz']['generate']:
    print('Reading csv files..')
    unlabeled_data = pd.read_csv('{}/unlabeled_data.csv'
                                 .format(NPZ_DEST),
                                 encoding='iso-8859-1')
    train_data = pd.read_csv('{}/train_data.csv'
                             .format(NPZ_DEST),
                             encoding='iso-8859-1')
    test_data = pd.read_csv('{}/test_data.csv'
                            .format(NPZ_DEST),
                            encoding='iso-8859-1')

    print('Generating feature vectors..')
    unlabeled_feat, _ = preprocess.count_vectorizer([unlabeled_data['email']])
    train_feat, test_feat = preprocess.count_vectorizer([
        train_data['email'], test_data['email']
    ])

    print('Exporting npz files inside {}..'.format(NPZ_DEST))
    np.savez('{}/unlabeled_feature'.format(NPZ_DEST),
             X=unlabeled_feat)
    np.savez('{}/train_feature'.format(NPZ_DEST),
             X=train_feat, Y=train_data['class'].values)
    np.savez('{}/test_feature'.format(NPZ_DEST),
             X=test_feat, Y=test_data['class'].values)

print('Building model..')
sda = StackedDenoisingAutoEncoder(
    batch_size=CONFIG['model']['batch_size'],
    classes=CONFIG['model']['classes'],
    epochs=CONFIG['model']['epochs'],
    n_folds=CONFIG['model']['n_folds'],
    hidden_layers=CONFIG['model']['hidden_layers'],
    noise_layers=CONFIG['model']['noise_layers'],
)
model = sda.build_sda()

model.add(sda.build_finetune())

model.compile(loss='categorical_crossentropy',
              optimizer=SGD())

X_train, Y_train = sda.dataset['train_data']
X_test, Y_test, Y_true = sda.dataset['train_data']

print('Finetuning the model..')
history = model.fit(
    X_train, Y_train, batch_size=sda.batch_size,
    nb_epoch=sda.epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
)

print('Evaluating model..')
y_pred = model.predict_classes(X_test)
print(Y_test[:5])
print(y_pred[:5])
accuracy = accuracy_score(Y_true, y_pred)
precision, recall, f1_score, _ = \
    precision_recall_fscore_support(Y_test, y_pred)

print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1 score: {}'.format(f1_score))

print('Saving config results inside experiments/{}_exp/ ..'
      .format(CONFIG['id']))
exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())
model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/config.json'.format(exp_dir), 'w') as f:
    json.dump(CONFIG, f)

print('Done!')
