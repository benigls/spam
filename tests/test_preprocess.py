#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from preprocess import PreProcess
from common import params


class TestPreProcess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.preprocess = PreProcess(
            params.DATASET_PATH,
            params.DATASET_SUBDIRS,
        )

    def tearDown(self):
        pass

    def test_preprocess_instance(self):
        """
        Test if preprocess is creating a instance.
        """
        self.assertIsInstance(self.preprocess, PreProcess)
