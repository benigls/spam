#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import json
import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (precision_score, recall_score, auc,
                             f1_score, accuracy_score, roc_curve)

from spam.common import utils
from spam.preprocess import preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder, LossHistory


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


start_time = timeit.default_timer()

np.random.seed(1337)

CONFIG, CONFIG_FILENAME = parse_config()
if not CONFIG:
    sys.exit()

NPZ_DEST = CONFIG['npz']['dest']
CSV_DEST = CONFIG['csv']['dest']
TOKENIZER = None

if CONFIG['csv']['generate']:
    file_path_list = utils.get_file_path_list(
        utils.dataset_meta(CONFIG['dataset']))

    # transform list of tuple into two list
    # e.g. [('/path/to/file', 'spam')] ==> ['path/to/file'], ['spam']
    paths, labels = zip(*file_path_list)

    # generate panda dataframes and export it to csv
    print('\n{}\n'.format('-' * 50))
    print('Generating dataframe..')
    dataset = pd.DataFrame(**utils.df_params(paths, labels))

    print('\nExporting dataframe into a csv file inside {}'
          .format(CSV_DEST))
    dataset.to_csv('{}/dataset.csv'.format(CSV_DEST))

if CONFIG['npz']['generate']:
    print('\n{}\n'.format('-' * 50))
    print('Reading csv files..')
    dataset = pd.read_csv('{}/dataset.csv'
                          .format(CSV_DEST),
                          encoding='iso-8859-1')

    print('Spliting the dataset..')
    x_unlabel, (x_train, y_train), (x_test, y_test) = \
        utils.split_dataset(dataset['body'].values,
                            dataset['label'].values)

    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')

    print('Generating feature matrix..')
    X_unlabel, X_train, X_test, tokenizer = \
        preprocess.feature_matrix(
            dataset=[x_unlabel, x_train, x_test, ],
            max_words=CONFIG['preprocess']['max_words'],
            max_len=CONFIG['preprocess']['max_len']
        )

    TOKENIZER = tokenizer

    print('Exporting npz files inside {}'.format(NPZ_DEST))
    np.savez('{}/unlabel.npz'.format(NPZ_DEST), X=X_unlabel)
    np.savez('{}/train.npz'.format(NPZ_DEST), X=X_train, y=y_train)
    np.savez('{}/test.npz'.format(NPZ_DEST), X=X_test, y=y_test)

print('\n{}\n'.format('-' * 50))
print('Building model..')

pretraining_history = LossHistory()
sda = StackedDenoisingAutoEncoder(
    batch_size=CONFIG['model']['batch_size'],
    classes=CONFIG['model']['classes'],
    epochs=CONFIG['model']['epochs'],
    n_folds=CONFIG['model']['n_folds'],
    hidden_layers=CONFIG['model']['hidden_layers'],
    noise_layers=CONFIG['model']['noise_layers'],
    history=pretraining_history,
)
model = sda.build_sda()

model.add(sda.build_finetune())

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
pretraining_history = sda.history

X_train, Y_train = sda.dataset['train']
X_test, Y_test, Y_true = sda.dataset['test']

print('\n{}\n'.format('-' * 50))
print('Finetuning the model..')

finetune_history = LossHistory()
model.fit(
    X_train, Y_train, batch_size=sda.batch_size,
    nb_epoch=sda.epochs, show_accuracy=True,
    validation_data=(X_test, Y_test), validation_split=0.1,
    callbacks=[finetune_history],
)

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
y_pred = model.predict_classes(X_test)

metrics = {}
metrics['accuracy'] = accuracy_score(Y_true, y_pred)
metrics['precision'] = precision_score(Y_true, y_pred)
metrics['recall'] = recall_score(Y_true, y_pred)
metrics['f1'] = f1_score(Y_true, y_pred)

false_positive_rate, true_positive_rate, _ = \
    roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

for key, value in metrics.items():
    print('{}: {}'.format(key, value))


exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])

print('\n{}\n'.format('-' * 50))
print('Saving config results inside {}'.format(exp_dir))

os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(model.to_json())

model.save_weights('{}/model_weights.hdf5'
                   .format(exp_dir), overwrite=True)

with open('{}/metrics.json'.format(exp_dir), 'w') as f:
    json.dump(metrics, f, indent=4)

with open('{}/vocabulary.json'.format(exp_dir), 'w') as f:
    vocabulary = [w for w in TOKENIZER.word_counts]
    json.dump(vocabulary, f)

plt.figure(1)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
         label='AUC = {}'.format(roc_auc))
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('{}/roc_curve.png'.format(exp_dir))

# TODO: add labels to loss history
plt.figure(2)
plt.title('Pretraining loss history')
plt.plot(pretraining_history.losses)
plt.savefig('{}/pretraining_loss.png'.format(exp_dir))

plt.figure(3)
plt.title('Finetune loss history')
plt.plot(finetune_history.losses)
plt.savefig('{}/finetune_loss.png'.format(exp_dir))

# print('Updating config id..')
# CONFIG['id'] += 1

with open(CONFIG_FILENAME, 'w+') as f:
    json.dump(CONFIG, f, indent=4)

print('Done!')

end_time = timeit.default_timer()
print('Run for %.2fm' % ((end_time - start_time) / 60.0))
