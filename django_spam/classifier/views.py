# -*- coding: utf-8 -*-

import json

import enchant

from django.views import generic
from django.http import HttpResponse

from nltk import tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from keras.models import model_from_config
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


PREFIX = '../experiments/exp_103'

with open('{}/model_structure.json'.format(PREFIX), 'r') as f:
    MODEL_STRUCTURE = json.load(f)

with open('{}/vocabulary.json'.format(PREFIX), 'r') as f:
    VOCABULARY = json.load(f)
    VOCABULARY = ' '.join(VOCABULARY)

MODEL = model_from_config(MODEL_STRUCTURE)
MODEL.load_weights('{}/model_weights.hdf5'.format(PREFIX))

TOKENIZER = Tokenizer(nb_words=1000)
TOKENIZER.fit_on_texts(VOCABULARY)


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

        X = TOKENIZER.texts_to_sequences([body, ])
        X = pad_sequences(X, maxlen=800, dtype='float64')

        y = MODEL.predict_classes(X)[0]
        Y = 'SPAM' if y else 'WEW'
        response_data = {'label': Y}

        return HttpResponse(
            json.dumps(response_data),
            content_type='application/json'
        )

    else:
        return HttpResponse(
            json.dumps({'Error': 'Only supports post requests.'}),
            content_type='application/json'
        )
