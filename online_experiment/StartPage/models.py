from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
import random
from django.core.mail import send_mail


author = 'Jonas Radbruch'

doc = """
Start Page for Online Experiments, to check whether Exp already full. Works only in Comb. with iban module! 

YOU NEEDS TO THE SET THE NUMBER OF PLAYERS IN settings.py
"""


class Constants(BaseConstants):
    name_in_url = 'hajime'
    players_per_group = None
    num_rounds = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):

    Email = models.EmailField()

    user_agent = models.CharField()

    window_height = models.CharField()

    window_width = models.CharField()

