#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from common import utils


class TestUtils(unittest.TestCase):
    """
    A test class that tests utility functions.
    """
    def setUp(self):
        self.dataset_meta = {}

    def tearDown(self):
        self.dataset_meta = None

    def test_get_file_path_list(self):
        """
        Test if utils.get_file_path_list can return a correct list.
        """
        self.assertEqual(
            self.file_path_list,
            utils.get_file_path_list(self.dataset_meta)
        )
