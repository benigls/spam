#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from sklearn.cross_validation import train_test_split

from spam.preprocess import preprocess


def get_file_path_list(dataset_meta):
    """ A helper function that accepts the path of enron dataset
    and return all the email file paths with class.
    """
    email_file_path_list = []

    for subdir in dataset_meta['subdirs']:
        for file in os.listdir(subdir['spam_path']):
            email_file_path_list.append((
                os.path.join(subdir['spam_path'], file),
                file.split('.')[3],
            ))

        for file in os.listdir(subdir['ham_path']):
            email_file_path_list.append((
                os.path.join(subdir['ham_path'], file),
                file.split('.')[3],
            ))

    return email_file_path_list


def split_dataset(file_path_list, seed=0):
    """
    A helper function that accepts list of file paths
    and splits them into unlabeled, train, test sets.
    """
    # transform list of tuple into two list
    # e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
    path, classification = zip(*file_path_list)

    # split the data into labeled and unlabeled
    unlabeled_path, labeled_path, \
        _, labeled_class = train_test_split(
            path,
            classification,
            test_size=0.1,
            random_state=seed,
        )

    # split data into train and test data
    train_path, test_path, \
        train_class, test_class = train_test_split(
            labeled_path,
            labeled_class,
            test_size=0.2,
            random_state=seed,
        )

    return unlabeled_path, (train_path, train_class), \
        (test_path, test_class)


def df_params(paths, labels):
    """ Returns a dict as a parameter for the dataframe. """
    unlabel, ham, spam = -1, 0, 1
    data = {'email': [], 'class': [], }
    columns = ['email', 'class']

    for path, label in zip(paths, labels):
        email = preprocess.read_email(path)
        if email == '':
            continue

        data['email'].append(email)
        if label == 'spam':
            data['class'] = spam
        elif label == 'ham':
            data['class'] = ham
        else:
            data['class'] = unlabel

    return {'data': data, 'columns': columns, }
