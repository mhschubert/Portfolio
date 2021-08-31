from otree.api import Currency as c, currency_range
from . import models
from ._builtin import Page, WaitPage
from .models import Constants


#--------------------
class QuestionFiller(Page):

    def vars_for_template(self):
        return{
            "role": self.player.role,
        }

    def is_displayed(self):
        return self.round_number == 1


#--------------------
class vignette(Page):
    form_model = models.Player
    form_fields = ['vigAnswer']

    def vars_for_template(self):

        # type of question asked - 8 screen ; 4 each
        vigOrder = self.player.participant.vars['vigOrder']
        self.player.vigType = vigOrder[self.round_number - 1]

        if self.player.vigType == 'risk':
            # text of question asked  [list index][present/future - dependent on role ?randomize?]
            riskNo = self.player.participant.vars['riskNo']
            questIndex = self.player.participant.vars['questRisk'][riskNo]
            question = Constants.riskVignette[questIndex]
        elif self.player.vigType == 'time':
            # text of question asked  [list index][present/future - dependent on role ?randomize?]
            timeNo = self.player.participant.vars['timeNo']
            questIndex = self.player.participant.vars['questTime'][timeNo]
            question = Constants.timeVignette[questIndex]

        question = dict(question)
        self.player.vigText = question[self.player.role()]

        if self.round_number == 1 or self.round_number == 5:
            toggleYes = "in"
        else:
            toggleYes = ""

        return{
            "round" : self.round_number,
            "question" : self.player.vigText,  # draw question form randomized order
            "type": self.player.vigType,  # defines type of input
            "toggleYes" : toggleYes,
            "role": self.player.role,
        }

    def before_next_page(self):
        # increment vignette page counter by 1 dependent on type
        if self.player.vigType == 'risk':
            self.player.participant.vars['riskNo'] += 1
        elif self.player.vigType == 'time':
            self.player.participant.vars['timeNo'] += 1

        if self.round_number == 8:
            self.player.natVigAnswerOrder = self.player.fillInOrder()

class vignette2(Page):
    form_model = models.Player
    form_fields = ['natVigAnswer1',
                'natVigAnswer2',
                'natVigAnswer3',
                'natVigAnswer4',
                'natVigAnswer5',]

    def vars_for_template(self):
        return{
            'text1': self.player.natVigAnswerOrder[0],
            'text2': self.player.natVigAnswerOrder[1],
            'text3': self.player.natVigAnswerOrder[2],
            'text4': self.player.natVigAnswerOrder[3],
            'text5': self.player.natVigAnswerOrder[4],
        }

    def is_displayed(self):
        return self.round_number == Constants.num_rounds

#--------------------
class BoukenQuestions(Page):
    form_model = models.Player
    form_fields = ['qBoukenself',
                   'qBoukenBeer',
                   'qBoukenWine',
                   'qBoukenSpirit',
                   'qBoukenMix',
                   'qBoukenSmoke',
                   'qBoukenSmokeFreq',
                   'qBoukenSport',
                   'qBoukenFood']

    def is_displayed(self):
        return self.round_number == Constants.num_rounds

#questions on further information
class SubjectInfo(Page):
    form_model = models.Player
    form_fields = [
                    'qTongue',
                    'qSecondTongue',
                    'qPetBoolean',
                    'qPetKind',
                    'qage',
                   'qgender',
                   'qPostalCode',
                   'qStudyArea',
                   'qStudySubject',
                   'qMathGrade',
                   'qIncome',
                   'qParents',
                   'qMarried',
                   'qMarriedParents']

    def is_displayed(self):
        return self.round_number == Constants.num_rounds


page_sequence = [
    QuestionFiller,
    vignette,
    BoukenQuestions,
    vignette2,
    SubjectInfo,
]
