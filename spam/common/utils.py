#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def get_file_path_list(dataset_meta):
    """
    A helper function that accepts the path of enron dataset and
    return all the email file paths with class.
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