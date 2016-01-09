#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from sklearn.cross_validation import train_test_split

from spam.preprocess import preprocess


def dataset_meta(dataset_config):
    """ Add paths to dataset config.
    add the path of enron_dataset subdirs ham and spam
    e.g. enron_dataset/enron1/spam/ = `spam_path`
         enron_dataset/enron3/ham/ = `ham_path`
         enron_dataset/enron6/ = `path`
    """
    dataset = dict(dataset_config)
    for i, subdir in enumerate(dataset_config['subdirs']):
        dataset['subdirs'][i]['path'] = os.path.join(
            dataset['path'], subdir['name']
        )
        dataset['subdirs'][i]['ham_path'] = os.path.join(
            subdir['path'], 'ham'
        )
        dataset['subdirs'][i]['spam_path'] = os.path.join(
            subdir['path'], 'spam'
        )

    return dataset


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
    ham, spam = 0, 1
    data = {'label': [], 'subject': [], 'body': []}
    columns = ['label', 'subject', 'body']

    for path, label in zip(paths, labels):
        subject, body = preprocess.read_email(path)
        if body == '':
            continue

        data['body'].append(body)
        data['subject'].append(subject)
        if label == 'spam':
            data['label'].append(spam)
        elif label == 'ham':
            data['label'].append(ham)

    return {'data': data, 'columns': columns, }
