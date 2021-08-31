# -*- coding: utf-8 -*-
from __future__ import division

import random

from otree.common import Currency as c, currency_range

from . import views
from ._builtin import Bot
from .models import Constants


class PlayerBot(Bot):
    """Bot that plays one round"""

    def play_round(self):

        yield (views.BankAccount, {
            'iban': 'DE12345678901234567890',
            'bic': '10020000'
        })

        yield (views.Verification, {
            'verification': True,
        })

        yield (views.End)

