#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from spam.preprocess import Preprocess


class TestPreprocess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.fake_email = \
            'Subject: get that new car 8434\n' \
            'people nowthe weather or climate in any particular ' \
            'environment can change and affect what people eat ' \
            'and how much of it they are able to eat .'

        self.tokenize_email = [
            'Subject', ':', 'get', 'that', 'new', 'car',
            '8434', 'people', 'nowthe', 'weather', 'or', 'climate',
            'in', 'any', 'particular', 'environment', 'can',
            'change', 'and', 'affect', 'what', 'people', 'eat',
            'and', 'how', 'much', 'of', 'it', 'they', 'are',
            'able', 'to', 'eat', '.',
        ]

    def tearDown(self):
        self.preprocess = None
        self.fake_email = None
        self.tokenize_email = None

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
        self.assertEqual(
            self.tokenize_email,
            Preprocess.tokenize(self.fake_email)
        )

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
