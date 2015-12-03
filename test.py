#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os


class TestDataSet(unittest.TestCase):
    """
    Class for testing the enron dataset.
    """
    def setUp(self):
        self.dataset_dir = 'enron_dataset'

    def tearDown(self):
        self.dataset_dir = ''


if __name__ == '__main__':
    unittest.main()
