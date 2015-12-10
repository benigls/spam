#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cross_validation import train_test_split

from spam.common import DATASET_META
from spam.common.utils import get_file_path_list
from spam.preprocess import preprocess


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

unlabeled_data = pd.DataFrame(
    data=[preprocess.read_email(path) for path in unlabeled_path],
    columns=['email'],
)

train_data = pd.DataFrame(
    data={
        'email': [preprocess.read_email(path) for path in train_path],
        'class': [1 if cl == 'spam' else 0 for cl in train_class]
    },
    columns=['email', 'class'],
)

test_data = pd.DataFrame(
    data={
        'email': [preprocess.read_email(path) for path in test_path],
        'class': [1 if cl == 'spam' else 0 for cl in test_class]
    },
    columns=['email', 'class', 'class2'],
)

unlabeled_data.to_csv('unlabel_data.csv')
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
