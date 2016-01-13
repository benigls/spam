# -*- coding: utf-8 -*-

from django.views import generic


class HomePageView(generic.TemplateView):
    """ Homepage view. """
    template_name = 'classifier/home.html'
