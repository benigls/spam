#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
import pandas as pd

from keras.optimizers import SGD
from spam.common import DATASET_META
from spam.common.utils import get_file_path_list, split_dataset
from spam.preprocess import preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder


UNLABEL = -1
HAM = 0
SPAM = 1

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--all', action='store_true')
parser.add_argument('-c', '--csv', action='store_true')
parser.add_argument('-n', '--npz', action='store_true')
parser.add_argument('-m', '--model', action='store_true')
args = parser.parse_args()

if args.csv or args.all:
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

    print('\nExporting dataframes into a csv files inside data/csv/ ..')
    unlabeled_data.to_csv('data/csv/unlabeled_data.csv')
    train_data.to_csv('data/csv/train_data.csv')
    test_data.to_csv('data/csv/test_data.csv')

if args.npz or args.all:
    print('Reading csv files..')
    unlabeled_data = pd.read_csv('data/csv/unlabeled_data.csv',
                                 encoding='iso-8859-1')
    train_data = pd.read_csv('data/csv/train_data.csv',
                             encoding='iso-8859-1')
    test_data = pd.read_csv('data/csv/test_data.csv',
                            encoding='iso-8859-1')

    print('Generating feature vectors..')
    unlabeled_feat, _ = preprocess.count_vectorizer([unlabeled_data['email']])
    train_feat, test_feat = preprocess.count_vectorizer([
        train_data['email'], test_data['email']
    ])

    print('Exporting npz files inside data/npz/ ..')
    np.savez('data/npz/unlabeled_feature', X=unlabeled_feat)
    np.savez('data/npz/train_feature',
             X=train_feat, Y=train_data['class'].values)
    np.savez('data/npz/test_feature',
             X=test_feat, Y=test_data['class'].values)

if args.model or args.all:
    print('Building model..')
    sda = StackedDenoisingAutoEncoder(
        batch_size=128, classes=2, epochs=0, n_folds=4,
        hidden_layers=[5000, 3500, 2000, 500, ],
        noise_layers=[0.3, 0.2, 0.1, ],
    )
    model = sda.build_sda()

    model.add(sda.build_finetune())

    sgd = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    X_train, Y_train = sda.dataset['train_data']
    X_test, Y_test = sda.dataset['train_data']

    print('Finetuning the model..')
    model.fit(
        X_train, Y_train, batch_size=sda.batch_size,
        nb_epoch=sda.epochs, show_accuracy=True,
        validation_data=(X_test, Y_test), validation_split=0.1,
    )

    print('Evaluating model..')
    score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)

    print('Test score: {}'.format(score[0]))
    print('Test accuracy: {}'.format(score[1]))

    print('Saving model structure and weights..')
    # TODO: Handle exceptions
    # TODO: Update folder names
    os.mkdir('experiments/{}_epochs'.format(sda.epochs))
    open('experiments/{}_epochs/model_structure.json'
         .format(sda.epochs), 'w').write(model.to_json())
    model.save_weights('experiments/{}_epochs/model_weights.hdf5'
                       .format(sda.epochs), overwrite=True)

    print('Done!')
