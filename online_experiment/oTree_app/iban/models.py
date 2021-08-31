# -*- coding: utf-8 -*-
# <standard imports>
from __future__ import division

import random

import otree.models
from otree.db import models
from otree import widgets
from otree.common import Currency as c, currency_range, safe_json
from otree.constants import BaseConstants
from otree.models import BaseSubsession, BaseGroup, BasePlayer
# </standard imports>

# additional models import / requires "localflavor" to be installed
from localflavor.generic.models import IBANField, BICField

author = 'Felix Albrecht ,Thomas Graeber, Thorben Woelk'

doc = """
Standardized End Page
"""

class Constants(BaseConstants):
    name_in_url = 'endofexperiment'
    players_per_group = None
    num_rounds = 1

    super = "Felix Albrecht"
    supermail = "f.albrecht@uni-bonn.de"


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):


    iban = IBANField(verbose_name="IBAN")

    bic = BICField(verbose_name="BIC")

    bankname = models.CharField(verbose_name="Nachname:",max_length=50)

    bankvname = models.CharField(verbose_name="Vorname:", max_length=50)

    street = models.CharField(verbose_name="Straße / Hausnr.:", max_length=50)

    city = models.CharField(verbose_name="Stadt:", max_length=50)

    zipcode = models.IntegerField(verbose_name="PLZ:")

    user_agent = models.CharField()
    window_height = models.CharField()
    window_width = models.CharField()

    ibanmsgseen = models.PositiveIntegerField(default=0)

