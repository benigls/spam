# -*- coding: utf-8 -*-

from django.views import generic


class HomePageView(generic.TemplateView):
    """ Homepage view. """
    # TODO: Fix apps template
    template_name = 'home.html'
