from django.conf.urls import patterns, include, url
from otree.default_urls import urlpatterns

import redirect.views

urlpatterns.append(
    url(r'^redirect/$', 'redirect.views.get_redir',name="redirection",)
)
