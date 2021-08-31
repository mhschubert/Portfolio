from otree.api import Currency as c, currency_range
from . import models
from ._builtin import Page, WaitPage
from .models import Constants



# --- welcome pages

class welcome(Page):

    # increment page counter
    def before_next_page(self):
        self.player.current_page_index += 1

    def vars_for_template(self):
        return{
            "role" : self.player.role,
        }

# --- bomb game

class BombGame(Page):
    form_model = models.Player
    form_fields = ['bombgamestring','bomb']

    # increment page counter
    def before_next_page(self):
        self.player.current_page_index += 1
        # check if bomb was clicked
        packages = [int(x) for x in self.player.bombgamestring.split(',')]
        self.player.clickedbomb = self.player.bomb in packages

    def vars_for_template(self):
        return{
            "role" : self.player.role,
            "bombBoolean": self.player.clickedbomb,
        }

    def is_displayed(self):
        # --- order 1
        if self.player.id_in_group < 3 and self.player.current_page_index == 2:
            showIt = True
        # --- order 2
        elif self.player.id_in_group > 2 and self.player.current_page_index == 3:
            showIt = True
        else:
            showIt = False

        return showIt


class BombGameResult(Page):
    form_model = models.Player
    form_fields = ['bombgamePayoff']

    # increment page counter
    def before_next_page(self):
        self.player.current_page_index += 1
        # insure that it winnings are 0 if lost
        if self.player.clickedbomb == True:
            self.player.bombgamePayoff = 0
        self.participant.vars['bombtask'] = self.player.bombgamePayoff

    def vars_for_template(self):

        if self.player.clickedbomb == True:
            overImg = "loser"
        #double check needed; because of data base issues the string mustn't be empty, therefore 0 was chosen; zero in single character string only possible if no field was chosen
        elif str(self.player.bombgamestring)=='101':
            overImg = "boring"
        else:
            overImg = "winner"

        return{
            "bombgame" : self.player.bombgamestring,
            "bomb" : self.player.bomb,
            "role" : self.player.role,
            "overImg" : overImg,
        }

# --- Time scheme for all 4 order possibilities

class TimeScheme(Page):
    form_model = models.Player
    form_fields = ['payoffScheme','paymentDates']

    # increment page counter
    def before_next_page(self):
        self.player.current_page_index += 1

        self.participant.vars['timetask'] = self.player.paymentDates

    def vars_for_template(self):
        return{
            "role" : self.player.role,
        }

    def is_displayed(self):
        # --- order 1
        if self.player.id_in_group < 3 and self.player.current_page_index == 3:
            showIt = True
        # --- order 2
        elif self.player.id_in_group > 2 and self.player.current_page_index == 2:
            showIt = True
        else:
            showIt = False

        return showIt



    # --- page sequence for both orders 
page_sequence = [
    #welcome,
    # order 1
    BombGame,
    TimeScheme,
    # order 2 - tense is solved in games
    TimeScheme,
    BombGame,
    # bomb results
    BombGameResult,
]
