from otree.api import Currency as c, currency_range
from . import models
from ._builtin import Page, WaitPage
from .models import Constants


class StartPage(Page):

    form_fields = ['Email',
                   'window_height',
                   'window_width',
                   'user_agent',]
    form_model = models.Player

    def before_next_page(self):
        parti = self.request.build_absolute_uri(self.player.participant._start_url())
        self.request.session["otree"] = parti
        self.request.session.set_expiry(1209600) # timeout 2 weeks

page_sequence = [
    StartPage,
]
