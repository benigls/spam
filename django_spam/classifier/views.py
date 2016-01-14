# -*- coding: utf-8 -*-

import json

from django.views import generic
from django.http import HttpResponse

# from keras.model import model_from_config


class HomePageView(generic.TemplateView):
    """ Homepage view. """
    # TODO: Fix apps template
    template_name = 'home.html'


def classify(request):
    if request.method == 'POST':
        # TODO: connect to preprocess and clean the data
        #       produce vocubulary of words from the dataset.

        # model = model_from_config(
        #     '../experiments/100_exp/model_structure.json')

        # model.load_weights(
        #     '../experiments/100_exp/model_weights.hdf5')
        body = request.POST.get('body')
        num = ord(body[0])

        label = 'HAM' if num % 2 else 'SPAM'
        response_data = {'label': label}

        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({'Error': 'Only supports post requests.'}),
            content_type='application/json'
        )
