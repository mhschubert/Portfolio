# Training Biologically Inspired Algorithms - Ant Colony Optimization & Genetic Algorithm
### Uses: machine learning, multiprocessing; Backends: None - Implementation from scratch
This is just me paying around with biologically inspired algorithms to see how different factors influence the outcome.

In code there are two algorithms implemented:

## Ant Colony Optimization (ACO)
I implemented the three versions of the algorithm proposed in the paper
[Pan et al. (2020) "Pearson correlation coefficient-based pheromone refactoring mechanism for multi-colony ant colony optimization"](https://link.springer.com/article/10.1007%2Fs10489-020-01841-x):
- **ACS**: Merger of AS and Ant-Q - selection mechanism is based on a pseudo-random proportion rule.
- **Max-Min Ant System (MMAS)**: Dorigio et al. (1989) firs tptoposed this version. To make the algorithm stay near the shortest path, only that one is given additional weight.
- **Dynamic Guidance Mechanism ACO (DGMACO)**: Min-span tree and max-span tree based guidance solution to overcome local minima and urge faster convergence

All three are set up in a single colony. Moreover, I implemented the communication mechanism (PRACO) propesed in the paper. That mechanism enables the colonies to communicate their solutions to overcome local minima.


## Genetic Algorithm:
The second algorithm is a standard genetic alggorithm.
For the crossover, I implemented 3 versions:
- **Ordered Crossover** proposed by Davis et al. 1986.
- **Cycle Crossover** proposed in Oliver et al. 1987.
- **Cycle Crossover 2** proposed in [Hussain et al. 2017](https://www.hindawi.com/journals/cin/2017/7430125/). It overcomes the problem that normal cycle crossover may result in offsprings identical to the parent generation.

