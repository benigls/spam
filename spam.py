#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.cross_validation import train_test_split

from dataset_meta import DATASET_META

from spam.common.utils import get_file_path_list


file_path_list = get_file_path_list(DATASET_META)
path, classification = zip(*file_path_list)

unlabeled_path, labeled_path, \
    unlabeled_class, labeled_class = train_test_split(
        path,
        classification,
        test_size=0.1,
    )

print(len(unlabeled_path))
print(len(unlabeled_class))
print(len(labeled_path))
print(len(labeled_class))
