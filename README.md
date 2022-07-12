




I am currently busy with my thesis, then I will optimize the code. 

# Optimal liquidation of foreign profits with a model-based Reinforcement Learning algorithm

This repository is about the realisation of the article: L. Li, P.-A. Matt, and C. Heumann, “Optimal liquidation of foreign currencies when fx rates follow a generalised ornstein-uhlenbeck process”, Applied Intelligence, pp. 1–14, Apr. 2022.
overall diagram:
<img width="612" alt="image" src="https://user-images.githubusercontent.com/25768931/178139405-3835cc3f-843d-4bc8-bf6c-e905b88b4805.png">

The stochastic processes used in our algorithm:
<img width="1377" alt="image" src="https://user-images.githubusercontent.com/25768931/178138518-8345f78d-228b-4335-8458-267c36e59218.png">

## Abstract

In this article, we consider the case of a multinational company realizing profits in a country other than its base country.
The currencies used in the base and foreign countries are referred to as the domestic and foreign currencies respectively.
For its quarterly and yearly financial statements, the company transfers its profits from a foreign bank account to a domestic
bank account. Thus, the foreign currency liquidation task consists formally in exchanging over a period T a volume V of
cash in the foreign currency f for a maximum volume of cash in the domestic currency d. The foreign exchange (FX)
rate that prevails at time t is denoted Xd/f (t) and is defined as the worth of one unit of currency d in the currency f . We
assume in this article that the natural logarithm of the FX rate $x_t = log X_{d/f} (t)$ follows a discrete generalized Ornstein Uhlenbeck (OU) process, a process which generalizes the Brownian motion and mean-reverting processes. We also assume
minimum and maximum volume constraints on each transaction. Foreign currency liquidation exposes the multinational
company to financial risks and can have a significant impact on its final revenues, since FX rates are hard to predict and often
quite volatile. We introduce a Reinforcement Learning (RL) framework for finding the liquidation strategy that maximizes
the expected total revenue in the domestic currency. Despite the huge success of Deep Reinforcement Learning (DRL) in
various domains in the recent past, existing DRL algorithms perform sub-optimally in this task and the Stochastic Dynamic
Programming (SDP) algorithm – which yields the optimal strategy in the case of discrete state and action spaces – is rather
slow. Thus, we propose here a novel algorithm that addresses both issues. Using SDP, we first determine numerically the
optimal solution in the case where the state and decision variables are discrete. We analyse the structure of the computed
solution and derive an analytical formula for the optimal trading strategy in the general continuous case. Quasi-optimal
parameters of the analytical formula can then be obtained via grid search. This method, simply referred to as ”Estimated
Optimal Liquidation Strategy” (EOLS) is validated experimentally using the Euro as domestic currency and 3 foreign
currencies, namely USD (US Dollar), CNY(Chinese Yuan) and GBP(Great British Pound). We introduce a liquidation
optimality measure based on the gap between the average transaction rate captured by a strategy and the minimum rate
over the liquidation period. The metric is used to compare the performance of EOLS to the Time Weighted Average Price
(TWAP), SDP and the DRL algorithms Deep Q-Network (DQN) and Proximal Policy Optimization (PPO). The results show
that EOLS outperforms TWAP by 54%, and DQN and PPO by 15 − 27%. EOLS runs in average 20 times faster than DQN
and PPO. It has a performance on par with SDP but runs 44 times faster. EOLS is the first algorithm that utilizes a closed form solution of the SDP strategy to achieve quasi-optimal decisions in a liquidation task. Compared with state-of-the-art
DRL algorithms, it exhibits a simpler structure, superior performance and significantly reduced compute time, making EOLS
better suited in practice.

## Getting Started

### Dependencies

* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
