#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import fnmatch

from spam.common import params


class TestDataSet(unittest.TestCase):
    """
    Class for testing the enron dataset.
    """
    def setUp(self):
        self.dataset_path = params.DATASET_PATH
        self.dataset_subdirs = params.DATASET_SUBDIRS

        # add the path of enron_dataset subdirs ham and spam
        # e.g. enron_dataset/enron1/spam/ = `spam_path`
        #      enron_dataset/enron3/ham/ = `ham_path`
        #      enron_dataset/enron6/ = `path`
        for i in range(len(self.dataset_subdirs)):
            self.dataset_subdirs[i]['path'] = os.path.join(
                self.dataset_path, self.dataset_subdirs[i]['name']
            )
            self.dataset_subdirs[i]['ham_path'] = os.path.join(
                self.dataset_subdirs[i]['path'], 'ham'
            )
            self.dataset_subdirs[i]['spam_path'] = os.path.join(
                self.dataset_subdirs[i]['path'], 'spam'
            )

    def tearDown(self):
        self.dataset_path = ''
        self.dataset_subdirs = []

    def count_files(self, path):
        """
        Function that counts files with a .txt extension in a given path.
        """
        return len(fnmatch.filter(os.listdir(path), '*.txt'))

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
