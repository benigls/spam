# -*- coding: utf-8 -*-

import json

import enchant

from django.views import generic
from django.http import HttpResponse

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from keras.model import model_from_config


PREFIX = '../experiments/100_exp'
MODEL = model_from_config('{}/model_structure.json'.format(PREFIX))
MODEL.load_weights('{}/model_weights.hdf5'.forma(PREFIX))

with open('{}/vocabulary.json', 'r') as f:
    VOCABULARY = json.load(f)


def clean_text(text):
    """ A function that cleans text for keras tokenazation. """
    lemma = WordNetLemmatizer()
    dictionary = enchant.Dict('en_US')

    word_list = [w for w in tokenize.word_tokenize(text)
                 if w.isalnum()]

    return ' '.join([lemma.lemmatize(word) for word in word_list
                     if word not in stopwords.words('english')
                     if dictionary.check(word)])


class HomePageView(generic.TemplateView):
    """ Homepage view. """
    # TODO: Fix apps template
    template_name = 'home.html'


def classify(request):
    if request.method == 'POST':
        body = request.POST.get('body')
        body = clean_text(body)

    else:
        return HttpResponse(
            json.dumps({'Error': 'Only supports post requests.'}),
            content_type='application/json'
        )
