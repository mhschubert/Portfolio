# -*- coding: utf-8 -*-
from __future__ import division

from otree.common import Currency as c, currency_range, safe_json

from . import models
from ._builtin import Page, WaitPage
from .models import Constants


class Iban(Page):
    form_model = models.Player
    form_fields = ['iban',
                   'bic',
                   'bankvname',
                   'bankname',
                   'street',
                   'city',
                   'zipcode',
                   'user_agent',
                   'window_height',
                   'window_width',
                   'ibanmsgseen',
    ]
    def vars_for_template(self):
        return{
            'msgseen':self.player.ibanmsgseen if self.player.ibanmsgseen != None else "",
            'riban':self.player.iban if self.player.iban != None else "",
            'rbic':self.player.bic if self.player.bic != None else "",
            'rbankvname':self.player.bankvname if self.player.bankvname != None else "",
            'rbankname':self.player.bankname if self.player.bankname != None else "",
            'rstreet':self.player.street if self.player.street != None else "",
            'rcity':self.player.city if self.player.city != None else "",
            'rzipcode':self.player.zipcode if self.player.zipcode != None else "",
        }


class End(Page):
    pass


page_sequence = [
    Iban,
    End
]
