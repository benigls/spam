#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json

from collections import OrderedDict

import spam

PATHS = [
    'data/csv/enron1_clean_dataset.csv',
    'data/csv/enron2_clean_dataset.csv',
    'data/csv/enron3_clean_dataset.csv',
    'data/csv/enron4_clean_dataset.csv',
    'data/csv/enron5_clean_dataset.csv',
    'data/csv/enron6_clean_dataset.csv',
]

if __name__ == '__main__':
    for path in PATHS:
        with open(CONFIG_FILENAME, 'r') as f:
            CONFIG = json.load(f, object_pairs_hook=OrderedDict)

        CONFIG['preprocess']['params']['read_csv_filepath'] = path

        with open(CONFIG_FILENAME, 'w+') as f:
            json.dump(CONFIG, f, indent=4)

        spam()

        print('\n{}\n'.format('=' * 50))
