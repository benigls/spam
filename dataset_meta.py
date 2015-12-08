#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


DATASET_PATH = 'enron_dataset'
DATASET_SUBDIRS = [
    {
        'name': 'enron1',
        'total_count': 5172,
        'ham_count': 3672,
        'spam_count': 1500,
    },
    {
        'name': 'enron2',
        'total_count': 5857,
        'ham_count': 4361,
        'spam_count': 1496,
    },
    {
        'name': 'enron3',
        'total_count': 5512,
        'ham_count': 4012,
        'spam_count': 1500,
    },
    {
        'name': 'enron4',
        'total_count': 6000,
        'ham_count': 1500,
        'spam_count': 4500,
    },
    {
        'name': 'enron5',
        'total_count': 5175,
        'ham_count': 1500,
        'spam_count': 3675,
    },
    {
        'name': 'enron6',
        'total_count': 6000,
        'ham_count': 1500,
        'spam_count': 4500,
    },
]

# add the path of enron_dataset subdirs ham and spam
# e.g. enron_dataset/enron1/spam/ = `spam_path`
#      enron_dataset/enron3/ham/ = `ham_path`
#      enron_dataset/enron6/ = `path`
for i in range(len(DATASET_SUBDIRS)):
    DATASET_SUBDIRS[i]['path'] = os.path.join(
        DATASET_PATH, DATASET_SUBDIRS[i]['name']
    )
    DATASET_SUBDIRS[i]['ham_path'] = os.path.join(
        DATASET_SUBDIRS[i]['path'], 'ham'
    )
    DATASET_SUBDIRS[i]['spam_path'] = os.path.join(
        DATASET_SUBDIRS[i]['path'], 'spam'
    )

DATASET_META = {
    'path': DATASET_PATH,
    'subdirs': DATASET_SUBDIRS,
}
