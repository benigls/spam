# -*- coding: utf-8 -*-

from django.views import generic


class HomePage(generic.TemplateView):
    """ Homepage view. """
    template_name = 'classifier/home.html'
