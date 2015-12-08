#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


FAKE_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    'fake_dataset',
)

FAKE_DATASET_SUBDIR = [
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
    'subdirs': FAKE_DATASET_SUBDIR,
    'file_path_list': [
        os.path.join(
            FAKE_DATASET_SUBDIR[0]['ham_path'],
            '0001.1999-12-10.farmer.ham.txt'
        ),
        os.path.join(
            FAKE_DATASET_SUBDIR[0]['ham_path'],
            '0003.1999-12-14.farmer.ham.txt'
        ),
        os.path.join(
            FAKE_DATASET_SUBDIR[0]['spam_path'],
            '0017.2003-12-18.GP.spam.txt'
        ),
    ]
}
