#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A set of function that cleans the dataset
for machine learning process.
"""

import io
import sys

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


def clean_text(text):
    """
    A function that cleans text (regex, token, stop).
    """
    word_list = remove_stopwords(tokenizer(regex(text)))
    return ' '.join(word_list)


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


@static_vars(success=0, fail=0)
def read_email(path, clean=True):
    """
    A function that accepts file paths and return it's contents.
    """
    with io.open(path, 'r', encoding='cp1252') as file:
        try:
            content = ''.join(file.readlines())
            read_email.success += 1
        except UnicodeDecodeError:
            content = ''
            read_email.fail += 1

        sys.stdout.write("Success: {} \t".format(read_email.success))
        sys.stdout.write("Fail: {} \r".format(read_email.fail))
        sys.stdout.flush()

        file.close()

    return clean_text(content) if clean else content
