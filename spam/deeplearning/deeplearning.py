#!/usr/bin/env python
# -*- coding: utf-8 -*-


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
