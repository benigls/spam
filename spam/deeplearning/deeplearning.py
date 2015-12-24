#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils


class DeepLearning:
    """
    Class for deel learning methods.
    """
    def __init__(self, **kwargs):
        self.dataset = self._get_dataset()
        self.epochs = kwargs.pop('epochs', 0)
        self.batch_size = kwargs.pop('batch_size', 0)
        self.classes = kwargs.pop('classes', 1)
        self.hidden_layers = kwargs.pop('hidden_layers', None)
        self.noise_layers = kwargs.pop('noise_layers', None)
        self.n_folds = kwargs.pop('n_folds', 1)

        for key in kwargs.keys():
            print('Argument {} doesn\'t recognize.'.format(key))

    def _get_dataset(self):
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

    def build_sda(self):
        """ Build  Stack Denoising Autoencoder and perform a
        layer wise pre-training.
        """
        encoders = []
        input_data = np.copy(self._get_dataset()['unlabeled_data'])

        for i, (n_in, n_out) in enumerate(zip(
                self.hidden_layers[:-1], self.hidden_layers[1:]),
                start=1):
            print('Training layer {}: {} Layers -> {} Layers'.
                  format(i, n_in, n_out))

            # build the denoising autoencoder model structure
            ae = Sequential()

            # build the encoder with the gaussian noise
            encoder = containers.Sequential([
                GaussianNoise(self.noise_layers[i - 1],
                              input_shape=(n_in,)),
                Dense(input_dim=n_in, output_dim=n_out,
                      activation='sigmoid')
            ])

            # build the decoder
            decoder = containers.Sequential([
                Dense(input_dim=n_out, output_dim=n_in,
                      activation='sigmoid')
            ])

            # build the denoising autoencoder
            ae.add(AutoEncoder(
                encoder=encoder, decoder=decoder,
                output_reconstruction=False,
            ))

            ae.compile(loss='mean_squared_error', optimizer='rms')

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
