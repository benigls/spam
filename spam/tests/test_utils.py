#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import fake_dataset_meta
from spam.common import utils


class TestUtils(unittest.TestCase):
    """
    A test class that tests utility functions.
    """
    def setUp(self):
        self.fake_dataset_meta = fake_dataset_meta.FAKE_DATASET_META

    def tearDown(self):
        self.fake_dataset_meta = None

    def test_get_file_path_list(self):
        """
        Test if utils.get_file_path_list can return a correct list.
        """
        self.assertEqual(
            set(self.fake_dataset_meta['file_path_list']),
            set(utils.get_file_path_list(self.fake_dataset_meta))
        )
