#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def get_file_path_list(dataset_meta):
    """
    A helper function that accepts the path of enron dataset and
    return all the email file paths.
    """
    email_file_path_list = []

    for subdir in dataset_meta['subdirs']:
        for file in os.listdir(subdir['spam_path']):
            email_file_path_list.append(
                os.path.join(subdir['spam_path'], file)
            )

        for file in os.listdir(subdir['ham_path']):
            email_file_path_list.append(
                os.path.join(subdir['ham_path'], file)
            )


def split_dataset(file_path_list):
    """
    A helper function that accepts a list of file path and split it
    into training data and test data.
    """
    pass
