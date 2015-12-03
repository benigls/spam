#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os


class TestDataSet(unittest.TestCase):
    """
    Class for testing the enron dataset.
    """
    def setUp(self):
        self.dataset_path = 'enron_dataset'
        self.dataset_subdirs = ['enron', ]

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


if __name__ == '__main__':
    unittest.main()
