#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


FAKE_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'fake_dataset',
)

FAKE_DATASET_SUBDIRS = [
    {
        'name': 'enron1',
        'total_count': 3,
        'ham_count': 2,
        'spam_count': 1,
        'path': os.path.join(FAKE_DATASET_PATH, 'enron1'),
        'ham_path': os.path.join(
            FAKE_DATASET_PATH,
            'enron1',
            'ham'
        ),
        'spam_path': os.path.join(
            FAKE_DATASET_PATH,
            'enron1',
            'spam'
        ),
    },
]

FAKE_DATASET_META = {
    'path': FAKE_DATASET_PATH,
    'subdirs': FAKE_DATASET_SUBDIRS,
}
