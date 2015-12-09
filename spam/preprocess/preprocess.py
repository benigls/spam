#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from nltk import tokenize
from nltk.corpus import stopwords


class Preprocess:
    """
    A Class that cleans the dataset for machine learning process.
    """
    @staticmethod
    def regex(text):
        """
        A function that removes non-alphanumeric, -, _ characters
        and the word `Subject:`, and `re:` in text.
        """
        clean_text = re.sub('Subject:|re:', '', text)
        clean_text = re.sub('[^\w]+', ' ', clean_text)

        return clean_text

    @staticmethod
    def tokenize(text):
        """
        A function that splits a text.
        """
        return tokenize.word_tokenize(text)

    @staticmethod
    def stopwords(word_list):
        """
        A function that remove stopwords from a list of words.
        """
        return [word for word in word_list
                if word not in stopwords.words('english')]
