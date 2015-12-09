#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from nltk import tokenize


class Preprocess:
    """
    A Class that cleans the dataset for machine learning process.
    """
    @staticmethod
    def regex(text):
        """
        A function that removes non-alphanumeric, -, _ characters
        and the word `Subject:`, `re:` and `re ` in text.
        """
        clean_text = re.sub('Subject:|re:|re ', ' ', text)
        clean_text = re.sub('[^\w]+', ' ', clean_text)

        return clean_text

    @staticmethod
    def tokenize(text):
        """
        A function that splits a text.
        """
        return tokenize.word_tokenize(text)
