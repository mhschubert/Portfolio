# Behavioral Online Experiment based on oTree (Between-Subject / AB-Testing) - Optimized for Mobile Device Use

### Uses: Python, JavaScript, HTML, CSS; Backends: oTree, Heroku

Data must first be generated for it to then be analysed. That was also part of my job during my position as a scientist. In the following, I have one example of the experiments I programmed in order to test hypotheses and generate data.

The experiment focuses on *testing the "linguistic-savings hypothesis"* proposed in [Chen (2013)](https://www.aeaweb.org/articles?id=10.1257/aer.103.2.690).
It does so by leveraging a grammtical pecularity of the German langugage, 
namely that the future may be referenced using the grammatical present as well as the grammatical future tense.
That allows for a **framing experiment** implemented as a **between-subject design (A/B-testing)**.

The experiment is **programmed** using a mixture of **Java Script, HTML**, and **Python**. <br>
It is impelmented as an online experiment.<br>
It was designed and conducted before Amazon MTurk really was a thing - consequently it needed a Heroku server (excluded) in the background to communicate with.
Moreover, it is **optimized for mobile device use**.

Originally implemented in the beginning of 2017.

**Requirements**:
- JavaScript ES6 (2015)
- HTML5
- Python<=3.5
- otree<2.0.27 <- important they changed a lot in the backend afterwards.
- corresponding django version

## Screenshots of the Experiment
As probably not many are willing to go to the lengths required for running the oTree experiment - even as a simple demo -
I included a few screenshots below so you are able to see the structure and design of the experiment:

### Risk Aversion Task as BombGame ([Crosetto & Filippi, 2013](https://link.springer.com/article/10.1007/s11166-013-9170-z))
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/BombGame_1.png" width="500" /> 
 <img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/BombGameResolution.png" width="500" /> 
</p>


### Immediacy Task (Modified, Hidden Multiple Choice List with Implicit Switching Point)
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/TimeGame_1.png" width="500" /> 
 <img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/TimeGame_2.png" width="500" /> 
</p>

### Belief Elicitation Task with Vignette
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/risk_vignette.JPG" width="500" /> 
 <img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/time_vignette.JPG" width="500" /> 
</p>

### Paragraph Construction / Elicitation of Grammatical Preferences

<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/online_experiment/screenshots/Paragraph.png" width="500" />
</p>
