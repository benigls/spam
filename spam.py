#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split

from spam.common import DATASET_META
from spam.common.utils import get_file_path_list
from spam.preprocess import preprocess


UNLABEL = -1
HAM = 0
SPAM = 1

file_path_list = get_file_path_list(DATASET_META)

# transform list of tuple into two list
# e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
path, classification = zip(*file_path_list)

# split the data into unlabeled and labeled data
unlabeled_path, labeled_path, \
    _, labeled_class = train_test_split(
        path,
        classification,
        test_size=0.1,
        random_state=0,
    )

# split data into train and test data
train_path, test_path, \
    train_class, test_class = train_test_split(
        labeled_path,
        labeled_class,
        test_size=0.2,
        random_state=0,
    )

# generate panda dataframes and export it to csv
unlabeled_data = pd.DataFrame(
    data={
        'email': [preprocess.read_email(path) for path in unlabeled_path],
        'class': [UNLABEL for _ in range(len(unlabeled_path))]
    },
    columns=['email', 'class'],
)

train_data = pd.DataFrame(
    data={
        'email': [preprocess.read_email(path) for path in train_path],
        'class': [SPAM if cl == 'spam' else HAM for cl in train_class]
    },
    columns=['email', 'class'],
)

test_data = pd.DataFrame(
    data={
        'email': [preprocess.read_email(path) for path in test_path],
        'class': [SPAM if cl == 'spam' else HAM for cl in test_class]
    },
    columns=['email', 'class'],
)

unlabeled_data.to_csv('unlabel_data.csv')
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
