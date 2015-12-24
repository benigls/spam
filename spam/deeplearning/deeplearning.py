#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.utils import np_utils


class DeepLearning:
    """
    Class for deel learning methods.
    """
    def __init__(self, **kwargs):
        self.dataset = _get_dataset()
        self.epochs = kwargs.pop('epochs', 0)
        self.batch_size = kwargs.pop('batch_size', 0)
        self.classes = kwargs.pop('classes', 1)
        self.hidden_layers = kwargs.pop('hidden_layers', None)
        self.noise_layers = kwargs.pop('noise_layers', None)
        self.n_folds = kwargs.pop('n_folds', 1)

        for key in kwargs.keys():
            print('Argument {} doesn\'t recognize.'.format(key))

    def _get_dataset():
        """ Get dataset and unpack it. """
        unlabeled_train = np.load('unlabeled_feature.npz')['X']

        train_data = np.load('train_feature.npz')
        X_train, Y_train = train_data['X'], train_data['Y']

        test_data = np.load('test_feature.npz')
        X_test, Y_test = test_data['X'], test_data['Y']

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        Y_train = np_utils.to_categorical(Y_train, self.classes)
        Y_test = np_utils.to_categorical(Y_test, self.classes)

        return {'unlabeled_data': unlabeled_train,
                'train_data': (X_train, Y_train),
                'test_data': (X_test, Y_test), }
