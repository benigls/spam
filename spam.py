#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import json
import timeit

import numpy as np

from spam.common import utils
from spam.dataset import EnronDataset
from spam.preprocess import Preprocess
from spam.deeplearning import StackedDenoisingAutoEncoder

start_time = timeit.default_timer()

np.random.seed(1337)

CONFIG_FILENAME = 'config.json'

with open(CONFIG_FILENAME, 'r') as f:
    CONFIG = json.load(f)

if not CONFIG:
    print('Can\'t read config file.')
    sys.exit()

print('\n{}\n'.format('-' * 50))

if CONFIG['dataset']['generate']:
    print('Reading the dataset..')
    dataset = EnronDataset(path=CONFIG['dataset']['path'])

    enron_dataset = dataset.get_dataset()

    if CONFIG['dataset']['output']:
        print('Exporting the dataset..')
        dataset.to_csv(filepath=CONFIG['dataset']['filepath'])


print('Reading the dataset..')
if CONFIG['preprocess']['params']['read_csv']:
    preprocessor = Preprocess(**CONFIG['preprocess']['params'])
else:
    preprocessor = Preprocess(dataset=enron_dataset,
                              **CONFIG['preprocess']['params'])

if CONFIG['preprocess']['clean_dataset']:
    print('Cleaning the dataset..')
    preprocessor.clean_data()

if CONFIG['preprocess']['output_csv']:
    print('Exporting clean dataset..')
    preprocessor.dataset.to_csv(CONFIG['preprocess']['output_csv_filepath'])

print('Spliting the dataset..')
enron_dataset = preprocessor.dataset
enron_dataset = utils.split_dataset(x=enron_dataset['body'].values,
                                    y=enron_dataset['label'].values)

print('Transforming dataset into vectors and matrices..')
enron_dataset = preprocessor.transform(dataset=enron_dataset)
vocabulary = preprocessor.vocabulary

print('\n{}\n'.format('-' * 50))
print('Building model..')
sda = StackedDenoisingAutoEncoder(**CONFIG['model']['params'])

print('Pretraining model..')
pretraining_history = sda.pre_train(enron_dataset.unlabel)

print('Finetuning the model..')
finetune_history = sda.finetune(train_data=enron_dataset.train,
                                test_data=enron_dataset.test, )

print('\n{}\n'.format('-' * 50))
print('Evaluating model..')
metrics = sda.evaluate(dataset=enron_dataset)

for key, value in metrics.items():
    print('{}: {}'.format(key, value))

print('\n{}\n'.format('-' * 50))
exp_dir = 'experiments/exp_{}'.format(CONFIG['id'])

print('Saving config results inside {}'.format(exp_dir))
os.makedirs(exp_dir, exist_ok=True)

open('{}/model_structure.json'.format(exp_dir), 'w') \
    .write(sda.model.to_json())

sda.model.save_weights('{}/model_weights.hdf5'
                       .format(exp_dir), overwrite=True)

data_meta = utils.get_dataset_meta(data_meta=enron_dataset)

with open('{}/metrics.json'.format(exp_dir), 'w') as f:
    json.dump(metrics, f, indent=4)

with open('{}/data_meta.json'.format(exp_dir), 'w') as f:
    json.dump(data_meta, f, indent=4)

with open('{}/vocabulary.json'.format(exp_dir), 'w') as f:
    json.dump(vocabulary, f)

utils.plot_loss_history(data=pretraining_history,
                        title='Pretraining loss history',
                        name='pretraining_loss',
                        path=exp_dir, )

utils.plot_loss_history(data=finetune_history,
                        title='Finetune loss history',
                        name='finetune_loss',
                        path=exp_dir, )

# print('Updating config id..')
# CONFIG['id'] += 1

# with open(CONFIG_FILENAME, 'w+') as f:
#     json.dump(CONFIG, f, indent=4)

end_time = timeit.default_timer()

print('Done!')
print('Run for %.2fm' % ((end_time - start_time) / 60.0))
