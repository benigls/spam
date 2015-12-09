#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

from spam.preprocess import preprocess
from fake_dataset_meta import FAKE_DATASET_META as FAKE


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
        self.regex_email = ' get that new car 8434 people nowthe ' \
            'weather or climate in any particular environment can ' \
            'change and affect what people eat and how much of it ' \
            'they are able to eat '
        self.tokenize_email = [
            'get', 'that', 'new', 'car', '8434', 'people', 'nowthe',
            'weather', 'or', 'climate', 'in', 'any', 'particular',
            'environment', 'can', 'change', 'and', 'affect', 'what',
            'people', 'eat', 'and', 'how', 'much', 'of', 'it',
            'they', 'are', 'able', 'to', 'eat',
        ]
        self.stopwords = [
            'get', 'new', 'car', '8434', 'people', 'nowthe',
            'weather', 'climate', 'particular', 'environment',
            'change', 'affect', 'people', 'eat', 'much', 'able',
            'eat',
        ]

    def tearDown(self):
        self.fake_email = None
        self.regex_email = None
        self.tokenize_email = None
        self.stopwords = None

    def test_preprocess_read_email(self):
        """
        Test if preprocess can read email from the dataset.
        """
        self.assertEqual(
            self.fake_email,
            preprocess.read_email(FAKE['file_path_list'][2][0])
        )

    def test_preprocess_regex_email(self):
        """
        Test if preprocess regex can remove noise characters.
        """
        self.assertEqual(
            self.regex_email,
            preprocess.regex(self.fake_email)
        )

    def test_preprocess_tokenize_email(self):
        """
        Test if preprocess can tokenize email.
        """
        self.assertEqual(
            set(self.tokenize_email),
            set(preprocess.tokenizer(self.regex_email))
        )

    def test_preprocess_stopwords(self):
        """
        Test if preprocess can remove stopwords.
        """
        self.assertEqual(
            set(self.stopwords),
            set(preprocess.remove_stopwords(self.tokenize_email))
        )

    def test_preprocess_clean_email(self):
        """
        Test of preprocess can clean(regex, token, stop) a email.
        """
        self.assertEqual(
            ' '.join(self.stopwords),
            preprocess.clean(self.fake_email)
        )
