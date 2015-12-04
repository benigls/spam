#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os

import params


class TestDataSet(unittest.TestCase):
    """
    Class for testing the enron dataset.
    """
    def setUp(self):
        self.dataset_path = params.DATASET_PATH
        self.dataset_subdirs = params.DATASET_SUBDIRS

        for i in range(len(self.dataset_subdirs)):
            self.dataset_subdirs[i]['path'] = os.path.join(
                self.dataset_path, self.dataset_subdirs[i]['name']
            )

    def tearDown(self):
        self.dataset_path = ''
        self.dataset_subdirs = []

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
            self.assertTrue(os.path.exists(
                os.path.join(subdir['path'], 'spam')
            ))

    def test_dataset_ham_folder_exist(self):
        """
        Test if the dataset sub-directories has a ham folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.exists(
                os.path.join(subdir['path'], 'ham')
            ))

    def test_dataset_spam_is_folder(self):
        """
        Test if the dataset sub-directories spam folder is a folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.isdir(
                os.path.join(subdir['path'], 'spam')
            ))

    def test_dataset_ham_is_folder(self):
        """
        Test if the dataset sub-directories ham folder is a folder.
        """
        for subdir in self.dataset_subdirs:
            self.assertTrue(os.path.isdir(
                os.path.join(subdir['path'], 'ham')
            ))

if __name__ == '__main__':
    unittest.main()
