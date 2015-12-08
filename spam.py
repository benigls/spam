#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split

from dataset_meta import DATASET_META

from spam.common.utils import get_file_path_list
from spam.preprocess import Preprocess


file_path_list = get_file_path_list(DATASET_META)

# transform list of tuple into two list
# e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
path, classification = zip(*file_path_list)

# split the data into unlabeled labeled
unlabeled_path, labeled_path, \
    unlabeled_class, labeled_class = train_test_split(
        path,
        classification,
        test_size=0.1,
        random_state=0,
    )

# Preprocess
preprocess = Preprocess()
