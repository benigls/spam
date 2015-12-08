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
        """
        A function that get path of all email files.
        """
        email_path_list = []

        for subdir in self.dataset_subdirs:
            for file in os.listdir(subdir['spam_path']):
                email_path_list.append(
                    os.path.join(subdir['spam_path'], file)
                )

            for file in os.listdir(subdir['ham_path']):
                email_path_list.append(
                    os.path.join(subdir['ham_path'], file)
                )

        return email_path_list

    def open_email(self, email_path_list=[]):
        """
        A function that opens a list of email file path
        and save it in a list.
        """
        # emails = []

        # for email_path in email_path_list:
        # with open(email_path_list[1], 'r') as email:
        #     content = ''.join(email.readlines())
