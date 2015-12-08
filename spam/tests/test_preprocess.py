#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from fake_dataset_meta import FAKE_DATASET_META
from spam.preprocess import Preprocess


class TestPreprocess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.preprocess = Preprocess()

    def tearDown(self):
        self.preprocess = None

    def test_preprocess_instance(self):
        """
        Test if preprocess is creating a instance.
        """
        self.assertIsInstance(self.preprocess, Preprocess)

    def test_preprocess_read_email(self):
        """
        Test if preprocess can read email from the dataset.
        """
        pass

    def test_preprocess_regex_email(self):
        """
        Test if preprocess regex can remove non-alphanumeric
        characters and the word `Subject:` and replace it with a space.
        """
        pass

    def test_preprocess_tokenize_email(self):
        """
        Test if preprocess can tokenize email.
        """
        pass

    def test_preprocess_stopwords(self):
        """
        Test if preprocess can remove stopwords.
        """
        pass

    def test_preprocess_clean_email(self):
        """
        Test of preprocess can clean a email.
        This involves replacing characters via regex,
        tokenizing, and removing stopwords.
        """
        pass

    def test_preprocess_bag_of_words(self):
        """
        Test if preprocess can produces a correct bag-of-words.
        """
        pass
