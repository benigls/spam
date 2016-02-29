#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

from spam.common.collections import Dataset, Data


def split_dataset(x, y, seed=0):
    """ Split the dataset into unlabel, train, and test sets. """
    # split the data into label and unlabel
    x_unlabel, x_label, _, y_label = \
        train_test_split(
            x,
            y,
            test_size=0.1,
            random_state=seed,
        )

    # split data into train and test data
    x_train, x_test, y_train, y_test = \
        train_test_split(
            x_label,
            y_label,
            test_size=0.2,
            random_state=seed,
        )

    return Dataset(
        x_unlabel,
        Data(x_train, None, y_train),
        Data(x_test, None, y_test)
    )


def get_dataset_meta(self, dataset=None):
    """ Get the dataset meta. """
    data_meta = {}

    data_meta['unlabeled_count'] = len(dataset.unlabel)
    data_meta['labeled_count'] = \
        len(dataset.train.X) + len(dataset.test.X)

    data_meta['train_data'] = {}
    data_meta['test_data'] = {}

    data_meta['train_data']['spam_count'] = int(sum(dataset.train.y))
    data_meta['train_data']['ham_count'] = \
        int(len(dataset.train.y) - sum(dataset.train.y))
    data_meta['train_data']['total_count'] = \
        data_meta['train_data']['spam_count'] + \
        data_meta['train_data']['ham_count']

    data_meta['test_data']['spam_count'] = int(sum(dataset.test.y))
    data_meta['test_data']['ham_count'] = \
        int(len(dataset.test.y) - sum(dataset.test.y))
    data_meta['test_data']['total_count'] = \
        data_meta['test_data']['spam_count'] + \
        data_meta['test_data']['ham_count']

    return data_meta


def plot_loss_history(data=None, title=None, name=None, path=None):
    """ Plot and export loss history. """
    # TODO: add labels to loss history
    plt.title(title)
    plt.plot(data)
    plt.savefig('{}/{}.png'.format(path, name))
