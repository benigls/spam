#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import os
import shutil
from datetime import datetime

from spam.preprocess import PreProcess
from spam.common import params


class TestPreProcess(unittest.TestCase):
    """
    Class for testing the preprocces.
    """
    def setUp(self):
        self.preprocess = PreProcess(
            params.DATASET_PATH,
            params.DATASET_SUBDIRS,
        )

        self.dataset_path = os.path.join(
            os.path.dirname(__file__),
            'test_dataset',
        )
        self.dataset_subdirs = [
            {
                'name': 'enron1',
                'total_count': 5,
                'ham_count': 3,
                'spam_count': 2,
                'path': os.path.join(self.dataset_path, 'enron1'),
                'ham_path': os.path.join(
                    self.dataset_path,
                    'enron1',
                    'ham'
                ),
                'spam_path': os.path.join(
                    self.dataset_path,
                    'enron1',
                    'spam'
                ),
            },
            {
                'name': 'enron2',
                'total_count': 6,
                'ham_count': 2,
                'spam_count': 4,
                'path': os.path.join(self.dataset_path, 'enron2'),
                'ham_path': os.path.join(
                    self.dataset_path,
                    'enron2',
                    'ham'
                ),
                'spam_path': os.path.join(
                    self.dataset_path,
                    'enron2',
                    'spam'
                ),
            },
        ]

        # create test dataset dir
        self.mkdir(self.dataset_path)

        # get current date YYYY-MM-DD
        current_date = datetime.now().strftime('%Y-%m-%d')

        # create subdirs and email create files
        for subdirs in self.dataset_subdirs:
            # create subdirs
            self.mkdir(subdirs['path'])
            self.mkdir(subdirs['ham_path'])
            self.mkdir(subdirs['spam_path'])

            # create spam email files
            for id in range(1, subdirs['spam_count'] + 1):
                self.mkfile(os.path.join(
                    subdirs['spam_path'],
                    '{}.{}.TEST.spam.txt'.format(
                        str(id).zfill(4), current_date
                    )
                ))

            # create ham email files
            for id in range(1, subdirs['ham_count'] + 1):
                self.mkfile(os.path.join(
                    subdirs['ham_path'],
                    '{}.{}.TEST.ham.txt'.format(
                        str(id).zfill(4), current_date
                    )
                ))

    def tearDown(self):
        self.preprocess = None

        # remove test dataset if it does exist.
        if os.path.exists(self.dataset_path):
            shutil.rmtree(self.dataset_path)

    def mkdir(self, dir):
        """
        Check if the directory exist and create if it doesn't.
        """
        if not os.path.exists(dir):
            os.mkdir(dir)

    def mkfile(self, file_path):
        """
        Check if the file exist and create if it doesn't.
        """
        if not os.path.isfile(file_path):
            file = open(file_path, 'w')
            file.close()

    def test_preprocess_instance(self):
        """
        Test if preprocess is creating a instance.
        """
        self.assertIsInstance(self.preprocess, PreProcess)

    def test_preprocess_open_email(self):
        """
        Test if preprocess can open email from the dataset.
        """
        pass

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
