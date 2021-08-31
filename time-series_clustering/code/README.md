# Hyperparameter Analysis of Dynamic Time Warping & Partitional Clustering on Time-Series Data using R & HPC

This analysis focuses on how the choice of certain hyperparameters for dynamic time warping (dtw) and the clustering algorithm affects the outcome.
To test this, I simulate a datset using 5 known player types appearing in Publig-Goods Games (PGGs), which is a well-studied problem in behavioral research.

Then, I evaluate the algorithm's performance for separating the observed time-series into distinct clusters of types. 
The analysis is implemented in R and makes use of the package dtwclust. 
Moreover, as the simulations are computationally expensive the code is written in such a way that it may be used on a HPC system running a SLURM job scheduling system.
That is implemented as a task-array.

Scripts:
1. *dtw_configuration.R*: Main script first simulating the data for all groups and then clsutering it using different hyperparameters. Finally it comapres the outcome and saves the best result.
2. *extended_configuration_testing.R*: Follow-up scipt in case not only the overall best result should be saved but also the best result, given only a specific algorithm or only a specific subset of hyperparameters was used.
3. *evaluate_configration_testing.R*: Evaluates the outcome and saves the respective figures to disk.
4. *evaluate_extended_testing.R*: Evaluates the outcome of the extended testing and saves the figures to disk.
5. *utils/*: helper and support scripts for the analysis
6. *jobscripts/*: Jobscripts to run the configuration testing on a HPC system using a SLURM job scheduler

## Outcome

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
