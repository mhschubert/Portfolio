# Using Symbolic Regression & Grammatical Evolution to Find a Meaningful Number of Clusters in R

The goal was to reduce the number of clusters capturing observed behavior in the form of time-series.
As the same individual may exhibit different behavioral patterns when exposed to different environments,
the number of clusters does not reflect the number of behavioral types but rather the (far) larger number of behavioral patterns.

Using symbolic regression and grammatical evolution, I try reducing the number of clusters by estimating a function which captures the behavioral type.<br>
Then clusters with similar or identical functions can be combined.

## Sample Results
The figure below shows the clusters for the data with a period length of 20. In green, the cluster centroids is shown.
The blue line shows the result, when forecasting the contributions using the functions estimated with symbolic regression.

Overall, the predicted outcomes are tracking the cluster centroids well. That speaks in favour of the overall appraoch.
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/symbolic-regression/figures/estimated_by_time_20-1.jpg" width="750" height="1050"/>
</p>

Now, we look at a specific subset of estimated functions - namely those using the own lagged contribution as well as the lagged contribution of the other group members as input.
As the input is two-dimensional, the plots are three dimensioal.
While we see that there is variance in the functions between clsuters (the planar plot differs markedly), there are overlaps (e.g., clusters 10, 20, and 23 or clusters 35 and 53).
Consequently, we can merge those clusters as the underlying funciton is very similar and thus the supposed behavioral type producing these differing patterns is so too.
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/symbolic-regression/figures/20_periods_others_own_1-1.jpg" width="700" height="700"/>
</p>
