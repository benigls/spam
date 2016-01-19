# -*- coding: utf-8 -*-

from django.conf.urls import url

from .views import HomePageView, classify


urlpatterns = [
    url(r'^$', HomePageView.as_view(), name='home'),
    url(r'classify/$', classify, name='classify'),
]
