#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
import pandas as pd

from spam.common import DATASET_META
from spam.common.utils import get_file_path_list, split_dataset
from spam.preprocess import preprocess


UNLABEL = -1
HAM = 0
SPAM = 1

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--read', action='store_true')
args = parser.parse_args()

if not args.read:
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

    print('Generating train dataframe..')
    train_data = pd.DataFrame(
        data={
            'email': [preprocess.read_email(path) for path in train_path],
            'class': [SPAM if cl == 'spam' else HAM for cl in train_class]
        },
        columns=['email', 'class'],
    )

    print('Generating test dataframe..')
    test_data = pd.DataFrame(
        data={
            'email': [preprocess.read_email(path) for path in test_path],
            'class': [SPAM if cl == 'spam' else HAM for cl in test_class]
        },
        columns=['email', 'class'],
    )

    print('Exporting dataframes into a csv files inside data/csv/ ..')
    unlabeled_data.to_csv('unlabeled_data.csv')
    train_data.to_csv('train_data.csv')
    test_data.to_csv('test_data.csv')
else:
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
