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
