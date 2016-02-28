#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
