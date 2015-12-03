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

    def tearDown(self):
        self.dataset_dir = ''

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
        subdir_path = ''
        for subdir in self.dataset_subdirs:
            subdir_path = os.path.join(self.dataset_path, subdir)
            self.assertTrue(os.path.exists(subdir_path))


if __name__ == '__main__':
    unittest.main()
