#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, AutoEncoder
from keras.layers.noise import GaussianNoise
from keras.callbacks import Callback

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve,
                             confusion_matrix, matthews_corrcoef)

from spam.common.exception import IllegalArgumentError


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        try:
            loss = logs.get('loss')
        except AttributeError:
            loss = None

        self.losses.append(loss)


class StackedDenoisingAutoEncoder:
    """ Class for deel learning methods. """
    def __init__(self, **kwargs):
        self.pretr_epochs = kwargs.pop('pretraining_epochs', 0)
        self.fine_epochs = kwargs.pop('finetune_epochs', 0)
        self.batch_size = kwargs.pop('batch_size', 0)
        self.classes = kwargs.pop('classes', 1)
        self.hidden_layers = kwargs.pop('hidden_layers', None)
        self.noise_layers = kwargs.pop('noise_layers', None)
        self.pretr_activ = kwargs.pop('pretraining_activation', 'sigmoid')
        self.pretr_opt = kwargs.pop('pretraining_optimizer', 'adadelta')
        self.pretr_loss = kwargs.pop('pretraining_loss', 'mse')
        self.fine_activ = kwargs.pop('finetune_activation', 'softmax')
        self.fine_opt = kwargs.pop('finetune_optimizer', 'adadelta')
        self.fine_loss = kwargs.pop('finetune_loss',
                                    'categorical_crossentropy')

        self.model = None

        for key, item in kwargs.items():
            raise IllegalArgumentError(
                'Keyword argument {} with a value of  {}, '
                'doesn\'t recognize.'.format(key, item))

    def pre_train(self, unlabel_data=None):
        """ Build Stack Denoising Autoencoder and perform a
        layer wise pre-training.
        """
        encoders = []
        noises = []
        pretraining_history = []

        input_data = np.copy(unlabel_data)

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
                      activation=self.pretr_activ),
            ])
            decoder = Dense(input_dim=n_out, output_dim=n_in,
                            activation=self.pretr_activ)

            # build the denoising autoencoder
            ae.add(AutoEncoder(
                encoder=encoder, decoder=decoder,
                output_reconstruction=False,
            ))
            ae.compile(loss=self.pretr_loss, optimizer=self.pretr_opt)

            temp_history = LossHistory()

            # train the denoising autoencoder and it will return
            # the encoded input as the input to the next layer.
            # it also store the loss history every epochs.
            ae.fit(input_data, input_data,
                   batch_size=self.batch_size,
                   nb_epoch=self.pretr_epochs,
                   callbacks=[temp_history],)

            pretraining_history += temp_history.losses
            encoders.append(ae.layers[0].encoder.layers[1])
            noises.append(ae.layers[0].encoder.layers[0])
            input_data = ae.predict(input_data)

        # merge denoising autoencoder layers
        model = Sequential()
        for encoder, noise in zip(encoders, noises):
            # model.add(noise)
            model.add(encoder)

        self.model = model

        return pretraining_history

    def finetune(self, train_data=None, test_data=None):
        """ Build the finetune layer for finetuning or
        supervise task and finetune the model.
        """
        self.model.add(Dense(input_dim=self.hidden_layers[-1],
                             output_dim=self.classes))

        finetune_history = LossHistory()

        self.model.compile(loss=self.fine_loss,
                           optimizer=self.fine_opt)

        self.model.fit(
            train_data.X, train_data.Y,
            batch_size=self.batch_size,
            nb_epoch=self.fine_epochs, show_accuracy=True,
            validation_data=(test_data.X, test_data.Y),
            validation_split=0.1,
            callbacks=[finetune_history],
        )

        return finetune_history.losses

    def evaluate(self, dataset=None):
        """ Evaluate the predicted labels and return the metrics. """
        metrics = {}
        y_pred = self.model.predict_classes(dataset.test.X)
        conf_matrix = confusion_matrix(dataset.test.y, y_pred)

        metrics['true_positive'], metrics['true_negative'], \
            metrics['false_positive'], metrics['false_negative'] = \
            int(conf_matrix[0][0]), int(conf_matrix[1][1]), \
            int(conf_matrix[0][1]), int(conf_matrix[1][0])

        false_positive_rate, true_positive_rate, _ = \
            roc_curve(dataset.test.y, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        metrics['accuracy'] = accuracy_score(dataset.test.y, y_pred)
        metrics['precision'] = precision_score(dataset.test.y, y_pred)
        metrics['recall'] = recall_score(dataset.test.y, y_pred)
        metrics['f1'] = f1_score(dataset.test.y, y_pred)
        metrics['mcc'] = matthews_corrcoef(dataset.test.y, y_pred)
        metrics['auc'] = roc_auc

        return metrics
