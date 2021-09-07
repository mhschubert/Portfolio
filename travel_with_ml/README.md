# Travelling Salesman Solution via Biologically Inspired Algorithms

### Uses: machine learning; Backends: None - implementation from scratch in python

I like to travel and I like to play around with data and machine learning. So this is may way of combining both:
Solving the problem of how to travel all world capital cities, the US state capitals, and Kisumu (Kenya) (my fianceé works there and it would be remiss of me not to visit). I use **biologically inspired** machine learning - it is basically "solving" a TSP problem.

Below you see the learning curves for the two algorithms:
<p float="left">
        <img src="https://github.com/mhschubert/Portfolio/blob/main/travel_with_ml/figures/cost_plot_Genetic.png" alt="drawing" width="500" height="500"/>
        <img src="https://github.com/mhschubert/Portfolio/blob/main/travel_with_ml/figures/cost_plot_Ant.png" alt="drawing" width="500" height="500"/>
</p>
As we can see, the ant colony algorithm outperforms the genetic one by at least one magnitude. Conseqeuntly, we use the latter for our travel plans:
<br><br><br>
<p float="left">
<img src="https://github.com/mhschubert/Portfolio/blob/main/travel_with_ml/figures/globe_travelpaco_states_kis_small.gif" width="500" height="500"/>
 <img src="https://github.com/mhschubert/Portfolio/blob/main/travel_with_ml/figures/lambert_travelpaco_states_kis.gif" width="500" height="400"/>
</p>
The solution is good, especially on the continental level. However, there are sudden jumps - to the Seychelles, for example. As a result I would say, that if I were to use that algorithm for travel
plans, I would chunk the great problem into smaller parts (i.e. continent-wise) with fixed entry-exit points. That should optimize the solution found!


