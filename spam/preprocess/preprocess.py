#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


class PreProcess:
    """
    A Class that cleans the dataset for machine learning process.
    """
    def __init__(self, dataset_path, dataset_subdirs):
        self.dataset_path = dataset_path
        self.dataset_subdirs = dataset_subdirs

    def get_email_path_list(self):
        email_path_list = []

        for subdir in self.dataset_subdirs:
            email_path_list += os.listdir(subdir['spam_path'])
            email_path_list += os.listdir(subdir['ham_path'])

        return email_path_list
