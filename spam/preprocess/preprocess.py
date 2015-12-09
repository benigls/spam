#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import re

from nltk import tokenize
from nltk.corpus import stopwords


def regex(text):
    """
    A function that removes non-alphanumeric, -, _ characters
    and the word `Subject:`, and `re:` in text.
    """
    clean_text = re.sub('Subject:|re:', '', text)
    clean_text = re.sub('[^\w]+', ' ', clean_text)

    return clean_text


def tokenizer(text):
    """
    A function that splits a text.
    """
    return tokenize.word_tokenize(text)


def remove_stopwords(word_list):
    """
    A function that remove stopwords from a list of words.
    """
    return [word for word in word_list
            if word not in stopwords.words('english')]


def clean(text):
    """
    A function that cleans text (regex, token, stop).
    """
    word_list = remove_stopwords(tokenizer(regex(text)))
    return ' '.join(word_list)


def read_email(path):
    """
    A function that accepts file paths and return it's contents.
    """
    with open(path, 'r') as file:
        content = ''.join(file.readlines())
        file.close()

    return content
