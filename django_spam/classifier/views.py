# -*- coding: utf-8 -*-

import json

from django.views import generic
from django.http import HttpResponse


class HomePageView(generic.TemplateView):
    """ Homepage view. """
    # TODO: Fix apps template
    template_name = 'home.html'


def classify(request):
    if request.method == 'POST':
        label = 1  # 1 = Spam 0 = Ham
        response_data = {'label': label}
        return HttpResponse(
            json.dumps(response_data),
            content_type="application/json"
        )
    else:
        return HttpResponse(
            json.dumps({"Error": "Only supports post requests."}),
            content_type="application/json"
        )
