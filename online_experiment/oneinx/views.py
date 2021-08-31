from otree.api import Currency as c, currency_range
from . import models
from ._builtin import Page, WaitPage
from .models import Constants


class Waiting(Page):

    def before_next_page(self):
        self.session.vars['winner'] += 1
        #self.player.finished = self.player.finisher()
        self.player.finished = self.session.vars['winner']
        self.player.iswinner = self.player.lottoWinner()
        # uncomment below before running real experiment
        ####self.player.iswinner = True

class Winner(Page):
    form_model = models.Player
    form_fields = ['qemail',
                   'qmailname']

    def vars_for_template(self):
        timepay = self.participant.vars['timetask'].split(',')
        bombpay = self.participant.vars['bombtask']
        return{
            "iswinner" : self.player.iswinner,
            "timepay" : timepay[0],
            "timedates" : timepay[1:],
            "bombpay" : bombpay ,
        }
    def before_next_page(self):
        self.player.mailMe(bombtask=str(self.participant.vars['bombtask']))

page_sequence = [
    Waiting,
    Winner
]
