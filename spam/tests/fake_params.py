#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'test_dataset',
)

DATASET_SUBDIRS = [
    {
        'name': 'enron1',
        'total_count': 5,
        'ham_count': 3,
        'spam_count': 2,
        'path': os.path.join(DATASET_PATH, 'enron1'),
        'ham_path': os.path.join(
            DATASET_PATH,
            'enron1',
            'ham'
        ),
        'spam_path': os.path.join(
            DATASET_PATH,
            'enron1',
            'spam'
        ),
    },
    {
        'name': 'enron2',
        'total_count': 6,
        'ham_count': 2,
        'spam_count': 4,
        'path': os.path.join(DATASET_PATH, 'enron2'),
        'ham_path': os.path.join(
            DATASET_PATH,
            'enron2',
            'ham'
        ),
        'spam_path': os.path.join(
            DATASET_PATH,
            'enron2',
            'spam'
        ),
    },
]
