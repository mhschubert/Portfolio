from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)


author = 'Marcel H. Schubert'

doc = """
Bomb Risk elicitation task
Time Preference elicitation task
"""


class Constants(BaseConstants):
    name_in_url = 'bouken'
    players_per_group = 4
    num_rounds = 1
    bombmulti = 0.20


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):

    #string of chosen fields in the bombGame
    bombgamestring = models.CharField(blank=True)
    #chosen PayoffScheme
    payoffScheme = models.IntegerField()
    #chosen PayoffScheme
    paymentDates = models.CharField()

    #earned payoff in the bombGame
    bombgamePayoff = models.FloatField()
    #field were the bomb was placed
    bomb = models.IntegerField()
    #field for the time spent on the bombgame
    bombtime = models.IntegerField()
    # winner in bombtime
    clickedbomb = models.BooleanField()
    #field for the time spent on the bombgameresult
    bombresulttime = models.IntegerField()
    #field for the time spent on the selection of the time preference task
    timetime = models.IntegerField()

    # page counter
    current_page_index = models.PositiveIntegerField(initial=1)

    def role(self):
        if self.id_in_group == 1 or self.id_in_group == 3:
            return 'present'
        else:
            return 'future'
