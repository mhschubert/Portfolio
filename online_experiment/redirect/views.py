# -*- coding: utf-8 -*-
from __future__ import division

from ._builtin import Page, WaitPage
from django.http import HttpResponse, HttpResponseRedirect

def get_redir(request):
    if request.session.get("otree"):
        cookieVal = request.session["otree"]
        return HttpResponseRedirect(cookieVal)
    else:
        if request.GET.get('join'):
            loc = request.GET['join']
            build = request.build_absolute_uri().split("redirect")
            location = build[0]+'join/'+loc+'/'
            return HttpResponseRedirect(location)
        else:
            return HttpResponse("Website unknown!")

class MyPage(Page):
    pass

   
page_sequence = [
    MyPage
]
