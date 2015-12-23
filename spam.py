#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
# from sklearn.cross_validation import train_test_split

# from spam.common import DATASET_META
# from spam.common.utils import get_file_path_list
from spam.preprocess import preprocess


# UNLABEL = -1
# HAM = 0
# SPAM = 1

# file_path_list = get_file_path_list(DATASET_META)

# # transform list of tuple into two list
# # e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
# path, classification = zip(*file_path_list)

# # split the data into unlabeled and labeled data
# print('spliting the dataset')
# unlabeled_path, labeled_path, \
#     _, labeled_class = train_test_split(
#         path,
#         classification,
#         test_size=0.1,
#         random_state=0,
#     )

# # split data into train and test data
# train_path, test_path, \
#     train_class, test_class = train_test_split(
#         labeled_path,
#         labeled_class,
#         test_size=0.2,
#         random_state=0,
#     )

# # generate panda dataframes and export it to csv
# print('generating unlabeled dataframe')
# unlabeled_data = pd.DataFrame(
#     data={
#         'email': [preprocess.read_email(path) for path in unlabeled_path],
#         'class': [UNLABEL for _ in range(len(unlabeled_path))]
#     },
#     columns=['email', 'class'],
# )

# print('generating train dataframe')
# train_data = pd.DataFrame(
#     data={
#         'email': [preprocess.read_email(path) for path in train_path],
#         'class': [SPAM if cl == 'spam' else HAM for cl in train_class]
#     },
#     columns=['email', 'class'],
# )

# print('generating test dataframe')
# test_data = pd.DataFrame(
#     data={
#         'email': [preprocess.read_email(path) for path in test_path],
#         'class': [SPAM if cl == 'spam' else HAM for cl in test_class]
#     },
#     columns=['email', 'class'],
# )

# # export dataframes to csv
# unlabeled_data.to_csv('unlabeled_data.csv')
# train_data.to_csv('train_data.csv')
# test_data.to_csv('test_data.csv')

# read_csv files
unlabeled_data = pd.read_csv('unlabeled_data.csv', encoding='iso-8859-1')
train_data = pd.read_csv('train_data.csv', encoding='iso-8859-1')
test_data = pd.read_csv('test_data.csv', encoding='iso-8859-1')

# generate feature vector
unlabeled_feat, _ = preprocess.count_vectorizer([unlabeled_data['email']])
train_feat, test_feat = preprocess.count_vectorizer([
    train_data['email'], test_data['email']
])

# save feature vector as .npz file
np.savez('unlabeled_feature.npy', X=unlabeled_feat)
np.savez('train_feature.npy', X=train_feat, y=train_data['class'].values)
np.savez('test_feature.npy', X=test_feat, y=test_data['class'].values)
