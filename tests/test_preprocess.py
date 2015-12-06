#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from preprocess import PreProcess


class TestPreProcess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.preprocess = PreProcess()

    def tearDown(self):
        pass

    def test_preprocess_instance(self):
        """
        Test if preprocess is creating a instance.
        """
        self.assertIsInstance(self.preprocess, PreProcess)
