#!/usr/bin/sh

# TODO: Refactor this garbage code

dataset_name="enron_dataset"

mkdir $dataset_name

cd $dataset_name

wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz"
wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz"
wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron3.tar.gz"
wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron4.tar.gz"
wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron5.tar.gz"
wget "http://www.aueb.gr/users/ion/data/enron-spam/preprocessed/enron6.tar.gz"

tar -xvzf enron1.tar.gz
tar -xvzf enron2.tar.gz
tar -xvzf enron3.tar.gz
tar -xvzf enron4.tar.gz
tar -xvzf enron5.tar.gz
tar -xvzf enron6.tar.gz

rm -rf enron1.tar.gz
rm -rf enron2.tar.gz
rm -rf enron3.tar.gz
rm -rf enron4.tar.gz
rm -rf enron5.tar.gz
rm -rf enron6.tar.gz

unset $dataset_name
