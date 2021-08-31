# Using Symbolic Regression & Grammatical Evolution for Estimating Functions to Describe Clusters

### Uses: unsupervised learning, symbolic regression, evolutionary learning; backend: R

In this project, I use symbolic regression via gramamtical evolution to find mathemtical functions.
The goal is to find a smaller number of functions describing a larger number of clusters so that the number of meaningful clusters may be compressed to a sensbile number.

The data I use is observational data from lab experiments implementing the Public-Goods game (PGG) prevalent in behavioral research.
In the PGG literature, there are five theorized behavioral programs: altruists, conditional cooperators, far-sighted free-riders, hump-shaped contributors, and short-sighted free-riders.

Given those five types and a group size of four, there are 35 possible combinations for three players.
That means that a fourth player would be exposed to 35 possible environments.
If that player is sensitive to the choices of the others, her choices may exhibit 35 different patterns despite her type being constant.
This example does not even account for random errors or other outside influences, which may make the resultant patterns noisy.

Consequently, extracting a meanigful number of behavioral types from observed patterns is non-trivial.

*Note*: The data needed to run the code is missing as I am not licensed to publicly share it. If you need it, please do not hestitate to contact me.
