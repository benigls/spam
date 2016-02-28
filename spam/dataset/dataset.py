#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pandas as pd

from spam.common.exception import IllegalArgumentError


class EnronDataset:
    """ Enron dataset interface. """
    def __init__(self, **kwargs):
        self.path = kwargs.pop('path', None)
        self.dataset = None

        for key, item in kwargs.items():
            raise IllegalArgumentError(
                'Keyword argument {} with a value of  {}, '
                'doesn\'t recognize.'.format(key, item))

    def _is_valid_file(self, filename=None):
        """ Check if the filename is valid. """
        filename_list = filename.split('.')

        # TODO: Add more validation

        if len(filename_list) != 5:
            return False

        return True

    def _read_email(self, path):
        with open(path, 'r', encoding='iso-8859-1') as f:
            try:
                content = f.readlines()
                content.pop(0)
                body = ''.join(content)
            except UnicodeDecodeError:
                content = ''

        return body

    def get_dataset(self):
        """ Generate panda dataframe of body and label. """
        if self.dataset:
            return self.dataset

        ham, spam = 0, 1
        path_list = []
        dataset = []

        for dn, sub_dn, file_paths in os.walk(self.path):
            path_list += [os.path.join(dn, fp) for fp in file_paths
                          if self._is_valid_file(fp)]

        for i, path in enumerate(path_list):
            dataset.append((self._read_email(path),
                            spam if path.split('.')[3] == 'spam' else ham))

        body, label = zip(*dataset)
        self.dataset = pd.DataFrame(**{
            'data': {
                'body': body,
                'label': label
            },
            'columns': ['body', 'label'],
        })

        return self.dataset

    def to_csv(self, path=None, name=None):
        """ Export dataset into csv file. """
        # check if dataset is loaded if not get it.
        if not self.dataset:
            self.get_dataset()

        self.dataset.to_csv('{}.csv'.format(os.path.join(path, name)))
