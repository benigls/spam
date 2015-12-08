#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset_meta import DATASET_META

from spam.common.utils import get_file_path_list


file_path_list = get_file_path_list(DATASET_META)
print(file_path_list)
