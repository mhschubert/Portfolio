from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
import random
from django.core.mail import send_mail


author = 'Felix Albrecht, Marcel H. Schubert'

doc = """
Meant to choose one winner per interval.
E.g., '1 in 500' wins
"""


class Constants(BaseConstants):
    name_in_url = 'kuji'
    players_per_group = None
    num_rounds = 1

    # interval from which one winner is drawn
    winOne = 250  # >>> 1 IN 500 wins / 1 for each of 500 <<<
    num_winners = 5  # number of winners 

    # create equal chunks
    max_players = winOne * num_winners
    chunks = [x for x in range(0,max_players+1,winOne)]


class Subsession(BaseSubsession):

    def before_session_starts(self):

        self.session.vars["winner"] = 0

        for el in range(0,len(Constants.chunks)-1):
            # draw random integer between two Constants.chunk borders
            num_win = random.randint(Constants.chunks[el]+1,Constants.chunks[el+1])
            setattr(self, "winner"+str(el+1), num_win)

class Group(BaseGroup):
    pass


class Player(BasePlayer):

    iswinner = models.BooleanField(default=False) # True if is winner
    finished = models.IntegerField(initial=0)

    qemail = models.CharField(verbose_name="E-Mail:")
    qmailname = models.CharField(verbose_name="Name:")

    def mailMe(self,bombtask):
        if self.iswinner == True:
            # payment scheme
            dates = self.participant.vars['timetask'][6:].replace(',','\n')
            timepay = self.participant.vars['timetask'].split(',')[0]

            # create message body
            message = " Mail: " + str(self.qemail) \
                        + "\n Name: " + str(self.qmailname) \
                        + "\n\n Aus dem Spiel 'Meide die Bombe' erhalten Sie "+str(self.participant.vars['bombtask'])+" Euro" \
                        + "\n\n Aus dem Spiel 'Wähle die Zeit' erhalten Sie "+str(timepay)+" Euro zu folgenden Terminen:\n"+dates \
                        + "\n Die Auszahlungen aus dem Fragebogen werden wir Ihnen später mitteilen." \
                        + "\n\n Der Versuchsleiter wird sich umgehend bei Ihnen melden und diese E-Mail nochmals bestätigen."
            # send email
            send_mail('Bonn EconLab - Bestätigung: Sie wurden ausgelost', message, 'bonnlabexperiment@gmail.com',
                      [str(self.qemail),'bonnlabexperiment@gmail.com'],
                      fail_silently=False,
                      auth_user="bonnlabexperiment@gmail.com",
                      auth_password="grammardiscounting")



    ## additional participant counter in case users don't complete the exp
    #def finisher(self):
    #    finishedOther = []
    #    player = Player.objects.raw('SELECT finished, id FROM oneinx_player WHERE session_id = '+str(self.session.id)+' AND finished = (SELECT MAX(finished) FROM oneinx_player)')
    #    for p in player:
    #        finishedOther.append(p)
    #    if(len(finishedOther) == 0):
    #        return 1
    #    else:
    #        return finishedOther[0].finished + 1

    def lottoWinner(self):
        for i in range(0,Constants.num_winners):
            winnerIds = getattr(self.subsession, "winner"+str(i+1))
            finisherId = self.finished
            if finisherId == winnerIds:
                return True

# ----------------------------------------------------
# functions not in player class - executed on database reset

# create the necessary number of winner fields in subsession table
# dependent on values in Constants class
for el in range(0,len(Constants.chunks)-1):
        # create field in player class with winner IDs
        Subsession.add_to_class("winner"+str(el+1),
                            models.PositiveIntegerField(initial=0))

