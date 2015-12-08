#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk import tokenize


class Preprocess:
    """
    A Class that cleans the dataset for machine learning process.
    """
    def tokenize(self, text):
        """
        A function that splits a text.
        """
        return tokenize.word_tokenize(text)
