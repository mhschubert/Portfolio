from otree.api import (
    models, widgets, BaseConstants, BaseSubsession, BaseGroup, BasePlayer,
    Currency as c, currency_range
)
import csv
import random


author = 'Marcel H. Schubert, Felix Albrecht'

doc = """
Vignette for perception of difference in time and risk
when varying German grammar in present and future tense.
"""


class Constants(BaseConstants):
    name_in_url = 'questionnaire'
    players_per_group = 4
    num_rounds = 8
    num_vigs = 4  # number of vignettes for risk and time; # for each
    dev = 5 ##deviation of average in risk

    vignetteOrder = [['risk','risk','risk','risk','time','time','time','time'],['time','time','time','time','risk','risk','risk','risk']]

    with open('questionnaire/time_vignette.csv', encoding='utf-8') as f:
        timeVignette = list(csv.DictReader(f))

    with open('questionnaire/risk_vignette.csv', encoding='utf-8') as f:
        riskVignette = list(csv.DictReader(f))



class Subsession(BaseSubsession):

    def creating_session(self):

        for player in self.get_players():
            # draw list of random indexes for vignette of length of csvs
            timeQuests = [x for x in range(0,len(Constants.timeVignette))]
            riskQuests = [x for x in range(0,len(Constants.riskVignette))]
            random.shuffle(timeQuests)
            random.shuffle(riskQuests)
            player.participant.vars['questTime'] = timeQuests 
            player.participant.vars['questRisk'] = riskQuests 

            # vignette page counter variable
            player.participant.vars['timeNo'] = 0
            player.participant.vars['riskNo'] = 0

            vignette = Constants.vignetteOrder.copy()
            random.shuffle(vignette)
            player.participant.vars['vigOrder'] = vignette[0]
        # Note: could both be integrated into 3 level dictionary for sleekness
        # but then indexation would be a mess




class Group(BaseGroup):
    pass


class Player(BasePlayer):
 # risk
    qBoukenself = models.IntegerField(initial=None,
               choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
               verbose_name='Wie schätzen Sie sich persönlich ein:\
                             Sind Sie im Allgemeinen ein risikobereiter Mensch,\
                             oder versuchen Sie, Risiko zu vermeiden? (0: gar nicht risikobereit; 10 sehr risikobereit)',
               )

# Wie häufig trinken Sie die folgenden alkoholischen Getränke?
    qBoukenBeer = models.IntegerField(initial=None,
                    choices=[(1, u"regelmäßig"),
                            (2, u"ab und zu"),
                            (3, u"selten"),
                            (4, u"nie")],
                    verbose_name="Wie häufig pro Woche trinken Sie Bier",)

    qBoukenWine = models.IntegerField(initial=None,
                                     choices = [(1,u"regelmäßig"),
                                                (2,u"ab und zu"),
                                                (3,u"selten"),
                                                (4,u"nie")],
                                       verbose_name="Wie häufig pro Woche trinken Sie Wein",
                                    )

    qBoukenSpirit = models.IntegerField(initial=None,
                                     choices = [(1,u"regelmäßig"),
                                                (2,u"ab und zu"),
                                                (3,u"selten"),
                                                (4,u"nie")],
                                       verbose_name="Wie häufig pro Woche trinken Sie Spirituosen (Schnaps, Weinbrand, etc.)",
                                      )

    qBoukenMix = models.IntegerField(initial=None,
                                     choices = [(1, u"regelmäßig"),
                                                (2, u"ab und zu"),
                                                (3, u"selten"),
                                                (4, u"nie")],
                                       verbose_name="Wie häufig pro Woche trinken Sie Mixgetränke (Alkopops, Cocktails, etc.)")

    qBoukenSmoke = models.BooleanField(initial=None,
                                verbose_name= "Rauchen Sie gegenwärtig?",
                             widget=widgets.RadioSelectHorizontal())

    qBoukenSmokeFreq = models.IntegerField(
                        verbose_name="Wie viele Zigaretten rauchen Sie täglich?",
                        choices=range(1, 51),
                        blank=True,
                        initial=None)

    qBoukenSport = models.IntegerField(
                    initial=None,
                    choices=[(1, u"3 mal wöchentlich und mehr"),
                                (2, u"bis zu 1-2 mal wöchentlich "),
                                (3, u"maximimal 1 mal alle 2 Wochen"),
                                (4, u"nie")],
                    verbose_name="Wie oft betreiben Sie aktiv Sport, Fitness oder Gymnastik?")

    qBoukenFood = models.IntegerField(initial=None,
                                            choices=[
                                                (0, u"gar nicht"),
                                                (1, u"ein wenig"),
                                                (2, u"stark"),
                                                (3, u"sehr stark")],
                                            verbose_name="Inwieweit achten Sie auf gesundheitsbewusste Ernährung?")


# ----------- socio demographics ----------------------- #allgemeine Informationen

    qage = models.IntegerField(min=18, max=99, blank=True, verbose_name="Bitte geben Sie Ihr Alter an")

    qgender = models.IntegerField(choices = [(1, u"männlich"),(2, u"weiblich"), (3,u"keine Angabe")],
                                              verbose_name="Bitte geben Sie ihr Geschlecht an",
                                              widget=widgets.RadioSelectHorizontal(), blank=True)

    qStudyArea = models.IntegerField(verbose_name="Falls Sie studieren oder bereits arbeiten: Bitte geben Sie Ihr Studienfeld oder Arbeitsfeld an",
                                       choices=[
                                                [0, "Auf mich trifft diese Frage nicht zu"],
                                                [1, "Betriebswirtschaftslehre"],
                                                [2, "Biologie"],
                                                [3, "Chemie"],
                                                [4, "Informatik"],
                                                [5, "Jura"],
                                                [6, "Geographie"],
                                                [7, "Geschichte"],
                                                [8, "Linguistik"],
                                                [9, "Mathematik"],
                                                [10, "Pharmazie"],
                                                [11, "Physik"],
                                                [12, "Politologie"],
                                                [13, "Psychologie"],
                                                [14, "Religionswissenschaften"],
                                                [15, "Soziologie oder Erziehungswissenschaften"],
                                                [16, "Volkswirtschaftslehre"],
                                                [17, "Sonstiges"],
                                           ], blank=True)

    qStudySubject = models.CharField(initial=None, blank=True, 
                                      verbose_name="Falls zutreffend geben Sie bitte Ihr Studienfach oder Arbeitsumfeld an. Im Falle einer Promotion bitte Ihr Promotionsfeld ")

    qPostalCode = models.CharField(blank=True,
                                    verbose_name="Bitte geben Sie noch die Postleitzahl Ihres Heimatortes an, also von dem Ort, in dem Sie aufgewachsen sind")

    qMathGrade = models.FloatField(blank=True,
                                    verbose_name="Tragen Sie hier bitte Ihre letzte Durchschnittsnote in Mathematik ein. (Skala: 1.0-6.0)",
                                   min=1.0,
                                   max=6.0,)

    qIncome = models.IntegerField(choices=[(1, "weniger als 150€"), (2,"zwischen 150€ und unter 200€"),(3,"zwischen 200€ und unter 250€"),
                                            (4, "zwischen 250€ und unter 300€"), (5, "zwischen 300€ und unter 350€"), (6, "zwischen 350€ und unter 400€"),
                                            (7, "zwischen 400€ und unter 450€"), (8, "zwischen 450€ und unter 500€"), (9, "zwischen 500€ und unter 550€"),
                                            (10, "zwischen 550€ und unter 600€"), (11, "zwischen 600€ und unter 650€"), (12, "650€ und mehr€")],
                                            verbose_name="Wie viel Geld haben Sie monatlich (nach Abzug von Warmmiete, Versicherung und Lebensmitteln) zur freien Verfügung", blank=True)

    qParents = models.BooleanField(blank=True, 
                                    verbose_name="Haben Ihre Eltern einen akademischen Abschluss (Universität oder Fachhochschule)?"
                                    )

    qMarried = models.IntegerField(choices=[(1,"ledig"), (2,"verheiratet"), (3,"geschieden"),
                                            ],
                                            verbose_name="Bitte geben Sie Ihren Familienstand an",
                                              widget=widgets.RadioSelectHorizontal(), blank=True)

    qMarriedParents = models.IntegerField(choices=[(1,"ledig"), (2,"verheiratet"), (3,"geschieden")
                                            ],
                                            verbose_name="Bitte geben Sie den Familienstand des Elternteils/der Elternteile an, bei dem/denen Sie aufgewachsen sind",
                                            widget=widgets.RadioSelectHorizontal(), blank=True)

    qTongue = models.BooleanField(blank=True, verbose_name="Ist Deutsch Ihre Muttersprache",
                                  widget=widgets.RadioSelectHorizontal(),)

    qSecondTongue = models.PositiveIntegerField(blank=True, choices=range(0,100),
                                                verbose_name="In welchem Alter hatten Sie das erste Mal intensiven Kontakt mit einer Fremdsprache? (z.B.: bilingual aufgewachsen, spielerisches Erlernen in der KiTa, Grundschulklasse, etc.)")

    qPetBoolean = models.BooleanField(initial=None, blank=True, verbose_name="Haben Sie ein Haustier")

    qPetKind = models.CharField(blank=True, verbose_name="Welches Haustier haben Sie")
#   From here the form fields for the vignette follow

    # time / risk
    vigType = models.CharField()
    # shown text from vignette.csv
    vigText = models.CharField()
    # supplied answer
    vigAnswer = models.PositiveIntegerField(min=0,max=100)
    # answers for what feels most best
    natVigAnswer1 = models.CharField(choices=[('present',1), ('future',2),]
                            ,widget=widgets.RadioSelectHorizontal())
    natVigAnswer2 = models.CharField(choices=[('present',1), ('future',2),]
                            ,widget=widgets.RadioSelectHorizontal())
    natVigAnswer3 = models.CharField(choices=[('present',1), ('future',2),]
                            ,widget=widgets.RadioSelectHorizontal())
    natVigAnswer4 = models.CharField(choices=[('present',1), ('future',2),]
                            ,widget=widgets.RadioSelectHorizontal())
    natVigAnswer5 = models.CharField(choices=[('present',1), ('future',2),]
                            ,widget=widgets.RadioSelectHorizontal())
    natVigAnswerOrder = models.CharField()

    # --- randomize orders of lego questions future / present
    def fillInOrder(self):
        fOrder = ''
        pf = ['p','f']
        for i in range(0,5):
            random.shuffle(pf)
            fOrder = fOrder+pf[0]
        return fOrder

    # --- player role
    def role(self):
        if self.id_in_group == 1 or self.id_in_group == 3:
            return 'present'
        else:
            return 'future'
