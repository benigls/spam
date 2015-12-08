#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import fnmatch
from datetime import datetime

from spam.common import DATASET_META


class TestDataSet(unittest.TestCase):
    """
    Class for testing the enron dataset.
    """
    def setUp(self):
        self.dataset_path = DATASET_META['path']
        self.dataset_subdirs = DATASET_META['subdir']

    def tearDown(self):
        self.dataset_path = ''
        self.dataset_subdirs = []

    def count_files(self, path):
        """
        A helper function that counts files with
        a .txt extension in a given path.
        """
        return len(fnmatch.filter(os.listdir(path), '*.txt'))

    def check_email_format(self, filenames):
        """
        A helper function that checks if the email file name has a
        correct format. IDENT.YYYY-MM-DD.OWNER.CLASS.txt
        """
        if list is type(filenames):
            pass
        elif str is type(filenames):
            filenames = list(filenames)
        else:
            print('This function only accept string and list of strings.')

        for filename in filenames:
            filename_array = filename.split('.')
            filename_array_len = len(filename_array)

            if filename_array_len != 5:
                print(
                    '{} has only {} info. It must be 5.\nFormat: '
                    'IDENT.YYYY-MM-DD.OWNER.CLASS.txt'.
                    format(filename, filename_array_len)
                )
                return False

            # checks if index 0 (IDENT) is a 4 digit number.
            elif not filename_array[0].isdigit() or \
                    len(filename_array[0]) is not 4:
                print(
                    '{} has a wrong IDENT. It must be a 4 digit'
                    'number.\nExample: `0001`, `1763`'.
                    format(filename)
                )
                return False

            # checks if index 1 (YYYY-MM-DD) is a valid date format.
            elif not datetime.strptime(filename_array[1], '%Y-%m-%d'):
                print(
                    '{} has a wrong date. It must be in this format '
                    '`YYYY-MM-DD`.\nExample: `2002-03-22`, 2012-10-11'.
                    format(filename)
                )
                return False

            # checks if index 3 (CLASS) is either spam or ham.
            elif filename_array[3] != 'ham' and \
                    filename_array[3] != 'spam':
                print(
                    '{} has a wrong class name. It must be either '
                    '`spam` or `ham`.'.format(filename)
                )
                return False

        return True

    def test_dataset_path_exist(self):
        """
        Test if the dataset path exist.
        """
        return self.assertTrue(os.path.exists(self.dataset_path))

    def test_dataset_path_is_folder(self):
        """
        Test if the dataset path is a directory.
        """
        return self.assertTrue(os.path.isdir(self.dataset_path))

    def test_dataset_subdirs_exist(self):
        """
        Test if the dataset sub-directories exist.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.exists(subdir['path']))

    def test_dataset_subdirs_is_folder(self):
        """
        Test if the dataset sub-directories is a folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.isdir(subdir['path']))

    def test_dataset_spam_folder_exist(self):
        """
        Test if the dataset sub-directories has a spam folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.exists(subdir['spam_path']))

    def test_dataset_ham_folder_exist(self):
        """
        Test if the dataset sub-directories has a ham folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.exists(subdir['ham_path']))

    def test_dataset_spam_is_folder(self):
        """
        Test if the dataset sub-directories spam folder is a folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.isdir(subdir['spam_path']))

    def test_dataset_ham_is_folder(self):
        """
        Test if the dataset sub-directories ham folder is a folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.isdir(subdir['ham_path']))

    def test_dataset_spam_count(self):
        """
        Test if the dataset spam counts are correct.
        """
        for subdir in self.dataset_subdirs:
            self.assertEqual(
                subdir['spam_count'],
                self.count_files(subdir['spam_path'])
            )

    def test_dataset_ham_count(self):
        """
        Test if the dataset ham counts are correct.
        """
        for subdir in self.dataset_subdirs:
            self.assertEqual(
                subdir['ham_count'],
                self.count_files(subdir['ham_path'])
            )

    def test_dataset_total_count(self):
        """
        Test if the dataset total counts are correct.
        """
        for subdir in self.dataset_subdirs:
            self.assertEqual(
                subdir['total_count'],
                self.count_files(subdir['ham_path']) +
                self.count_files(subdir['spam_path'])
            )

    def test_dataset_email_format(self):
        """
        Test if the dataset email name formats are correct.
        """
        for subdir in self.dataset_subdirs:
            paths = os.listdir(subdir['ham_path']) + \
                os.listdir(subdir['spam_path'])

            self.assertTrue(self.check_email_format(paths))
