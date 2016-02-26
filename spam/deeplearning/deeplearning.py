#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils


class IllegalArgumentError(ValueError):
    pass


class StackedDenoisingAutoEncoder:
    """ Class for deel learning methods. """
    def __init__(self, **kwargs):
        self.epochs = kwargs.pop('epochs', 0)
        self.batch_size = kwargs.pop('batch_size', 0)
        self.classes = kwargs.pop('classes', 1)
        self.hidden_layers = kwargs.pop('hidden_layers', None)
        self.noise_layers = kwargs.pop('noise_layers', None)
        self.pretr_activ = kwargs.pop('pretraining_activation', 'sigmoid')
        self.pretr_opt = kwargs.pop('pretraining_optimizer', 'adadelta')
        self.pretr_loss = kwargs.pop('pretraining_loss', 'mse')
        self.dataset = self.get_dataset()

        for key, item in kwargs.items():
            raise IllegalArgumentError(
                'Keyword argument {} with a value of  {}, '
                'doesn\'t recognize.'.format(key, item))

    def get_dataset(self):
        """ Get dataset and unpack it. """
        prefix = 'data/npz'

        X_unlabel = np.load('{}/unlabel.npz'.format(prefix))['X']

        train_data = np.load('{}/train.npz'.format(prefix))
        X_train, y_train = train_data['X'], train_data['y']

        test_data = np.load('{}/test.npz'.format(prefix))
        X_test, y_test = test_data['X'], test_data['y']

        Y_train = np_utils.to_categorical(y_train, self.classes)
        Y_true = np.asarray(y_test, dtype='int32')
        Y_test = np_utils.to_categorical(y_test, self.classes)

        return {'unlabel': X_unlabel,
                'train': (X_train, Y_train),
                'test': (X_test, Y_test, Y_true), }

    def build_sda(self):
        """ Build Stack Denoising Autoencoder and perform a
        layer wise pre-training.
        """
        encoders = []

        input_data = np.copy(self.dataset['unlabel'])

        for i, (n_in, n_out) in enumerate(zip(
                self.hidden_layers[:-1], self.hidden_layers[1:]),
                start=1):
            print('Training layer {}: {} Layers -> {} Layers'.
                  format(i, n_in, n_out))

            # build the denoising autoencoder model structure
            ae = Sequential()

            encoder = containers.Sequential([
                GaussianNoise(self.noise_layers[i - 1], input_shape=(n_in,)),
                Dense(input_dim=n_in, output_dim=n_out,
                      activation=self.pretr_activ, init='uniform'),
            ])
            decoder = Dense(input_dim=n_out, output_dim=n_in,
                            activation=self.pretr_activ)

            # build the denoising autoencoder
            ae.add(AutoEncoder(
                encoder=encoder, decoder=decoder,
                output_reconstruction=False,
            ))
            ae.compile(loss=self.pretr_loss, optimizer=self.pretr_opt)

            # train the denoising autoencoder and it will return
            # the encoded input as the input to the next layer.
            ae.fit(input_data, input_data,
                   batch_size=self.batch_size, nb_epoch=self.epochs)

            encoders.append(ae.layers[0].encoder)
            input_data = ae.predict(input_data)

        # merge denoising autoencoder layers
        model = Sequential()
        for encoder in encoders:
            model.add(encoder)

        return model

    def build_finetune(self, activation='softmax'):
        """ Build the finetune layer for finetuning or
        supervise task.
        """
        return Dense(input_dim=self.hidden_layers[-1],
                     output_dim=self.classes, activation=activation)

    def evaluate(self, Y, y):
        """ Evaluate the predicted labels and return metrics. """
        pass
