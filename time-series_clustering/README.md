# Using Dynamic Time Warping to Cluster Time Series of Group Interactions

In this project, I use time-series clustering with dynamic time warping on data of group interactions. 
The data used is closely relate to that of experiments implementing the Public-Goods game (PGG) prevalent in behavioral research.
However, as the goal is to asses the quality of the result, I use simulated data as it gives us a ground truth 

The goal is to find a good algorithmic configuration to partition that kind of data in a meaningful way.

*Note*: The data needed to run the code is missing as I am not licensed to publicly share it. If you need it, please do not hestitate to contact me.

The working paper for which I designed the code can be found here: [Engel, Hausladen, Schubert (2021): Charting the Type Space: The Case of Linear Public-Good Experiments](https://github.com/mhschubert/Portfolio/blob/main/time-series_clustering/pdf/Engel_Hausladen_Schubert_Charting.pdf) (authors are in alphabetical order)

## Parts of the Outcome

The following figure shows how the parameter alpha affects the optimal number of clusters for the respectively best algorithmic configurations (missing values means that confguration was never the best for the given k/number of clusters):
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/time-series_clustering/figures/rank_sum_within_configuration_n%3D6-1.jpg" width="1600" height="1600"/>
</p>

From that analysis, we concluded that partitional serves best for our purposes (high enough variance in the cluster evalution indices used and clear peaks an changes in the ranking of the clsutering result for different k).

As the goal of the analysis was to induce e mediuam amount of clusters (~30) as well as enough variation, I also looked at the distribution of the results given gamma:
From the overall analysis, we concluded that a gamma in the range of 0.0085 - 0.01 was optimal:

<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/time-series_clustering/figures/reevaluation_gamma_for_k-1.jpg" width="800" height="600"/>
</p>

The analysis was initially conducted for the working paper [Engel, Hausladen, Schubert (2021): Charting the Type Space: The Case of Linear Public-Good Experiments](https://github.com/mhschubert/Portfolio/blob/main/time-series_clustering/pdf/Engel_Hausladen_Schubert_Charting.pdf) (authors are in alphabetical order).
