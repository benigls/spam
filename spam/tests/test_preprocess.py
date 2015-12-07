#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os

import fake_params as fake
from spam.preprocess import PreProcess


class TestPreProcess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.preprocess = PreProcess(
            fake.DATASET_PATH,
            fake.DATASET_SUBDIRS,
        )
        self.dataset_path = fake.DATASET_PATH
        self.dataset_subdirs = fake.DATASET_SUBDIRS
        self.dataset_email_path_list = []
        self.email_list = []

        # get email path list
        for subdir in self.dataset_subdirs:
            for file in os.listdir(subdir['spam_path']):
                self.dataset_email_path_list.append(
                    os.path.join(subdir['spam_path'], file)
                )

            for file in os.listdir(subdir['ham_path']):
                self.dataset_email_path_list.append(
                    os.path.join(subdir['ham_path'], file)
                )

    def tearDown(self):
        self.preprocess = None
        self.dataset_path = None
        self.dataset_subdirs = None
        self.dataset_email_path_list = None
        self.email_list = None

    def test_preprocess_instance(self):
        """
        Test if preprocess is creating a instance.
        """
        self.assertIsInstance(self.preprocess, PreProcess)

    def test_preprocess_get_emails_path(self):
        """
        Test if preprocess can get the emails path.
        """
        self.assertEqual(
            self.dataset_email_path_list,
            self.preprocess.get_email_path_list()
        )

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
