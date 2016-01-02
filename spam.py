#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score, roc_curve,
                             auc)
from spam.common.utils import (get_file_path_list, split_dataset,
                               df_params, dataset_meta)

from spam.preprocess import preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder


def parse_config():
    """ Parses the args and return the config file in json. """
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', action='store')
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            config = json.load(f)

        return config, args.config
    except Exception as e:
        print('Error: {}'.format(e))
        return None, None


CONFIG, CONFIG_FILENAME = parse_config()
if not CONFIG:
    sys.exit()

NPZ_DEST = CONFIG['npz']['dest']
CSV_DEST = CONFIG['csv']['dest']

if CONFIG['csv']['generate']:
    file_path_list = get_file_path_list(dataset_meta(CONFIG['dataset']))

    print('\n{}\n'.format('-' * 50))
    print('Spliting the dataset..')
    unlabeled_path, (train_path, train_class), \
        (test_path, test_class) = split_dataset(file_path_list, seed=1337)

    # generate panda dataframes and export it to csv
    print('Generating unlabeled dataframe..')
    unlabeled_data = pd.DataFrame(**df_params(
        paths=unlabeled_path,
        labels=[None for _ in range(len(unlabeled_path))]))

    print('\nGenerating train dataframe..')
    train_data = pd.DataFrame(**df_params(
        paths=train_path, labels=train_class))

    print('\nGenerating test dataframe..')
    test_data = pd.DataFrame(**df_params(
        paths=test_path, labels=test_class))

    print('\nExporting dataframes into a csv files inside {} ..'
          .format(CSV_DEST))
    unlabeled_data.to_csv('{}/unlabeled_data.csv'.format(CSV_DEST))
    train_data.to_csv('{}/train_data.csv'.format(CSV_DEST))
    test_data.to_csv('{}/test_data.csv'.format(CSV_DEST))

if CONFIG['npz']['generate']:
    print('\n{}\n'.format('-' * 50))
    print('Reading csv files..')
    unlabeled_data = pd.read_csv('{}/unlabeled_data.csv'
                                 .format(CSV_DEST),
                                 encoding='iso-8859-1')
    train_data = pd.read_csv('{}/train_data.csv'
                             .format(CSV_DEST),
                             encoding='iso-8859-1')
    test_data = pd.read_csv('{}/test_data.csv'
                            .format(CSV_DEST),
                            encoding='iso-8859-1')

    print('Generating feature vectors..')
    unlabeled_feat, _ = preprocess.count_vectorizer(
        [unlabeled_data['email'], None],
        max_features=CONFIG['max_features']
    )
    train_feat, test_feat = preprocess.count_vectorizer([
        train_data['email'], test_data['email']
    ])

    print('Exporting npz files inside {}..'.format(NPZ_DEST))
    np.savez('{}/unlabeled_feature'.format(NPZ_DEST),
             X=unlabeled_feat)
    np.savez('{}/train_feature'.format(NPZ_DEST),
             X=train_feat, Y=train_data['class'].values)
    np.savez('{}/test_feature'.format(NPZ_DEST),
             X=test_feat, Y=test_data['class'].values)

print('\n{}\n'.format('-' * 50))
print('Building model..')
sda = StackedDenoisingAutoEncoder(
    batch_size=CONFIG['model']['batch_size'],
    classes=CONFIG['model']['classes'],
    epochs=CONFIG['model']['epochs'],
    n_folds=CONFIG['model']['n_folds'],
    hidden_layers=CONFIG['model']['hidden_layers'],
    noise_layers=CONFIG['model']['noise_layers'],
)
model = sda.build_sda()

model.add(sda.build_finetune())

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

X_train, Y_train = sda.dataset['train_data']
X_test, Y_test, Y_true = sda.dataset['test_data']

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')
history = model.fit(
    X_train, Y_train, batch_size=sda.batch_size,
    nb_epoch=sda.epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(X_test)

accuracy = accuracy_score(Y_true, y_pred)
precision = precision_score(Y_true, y_pred)
recall = recall_score(Y_true, y_pred)
f1 = f1_score(Y_true, y_pred)

false_positive_rate, true_positive_rate, _ = \
    roc_curve(Y_true, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

print(y_pred)
print(Y_true)

print('Accuracy: {}'.format(accuracy))
print('Precision: {}'.format(precision))
print('Recall: {}'.format(recall))
print('F1: {}'.format(f1))

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = {}'.format(roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print('\n{}\n'.format('-' * 50))
print('Saving config results inside experiments/{}_exp/ ..'
      .format(CONFIG['id']))
exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())
model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/config.json'.format(exp_dir), 'w') as f:
    json.dump(CONFIG, f)

plt.savefig('{}/roc_curve.png'.format(exp_dir))

print('Updating config id..')
CONFIG['id'] += 1

with open(CONFIG_FILENAME, 'w+') as f:
    json.dump(CONFIG, f)

print('Done!')
