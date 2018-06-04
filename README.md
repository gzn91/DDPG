# Generalizing DDPG

The implementation I used in my bachelor thesis. The goal of the project was to increase the stability of DDPG aswell as the ability to generalize to new tracks in the racing simluator TORCS.

This was acheived by using drop-out layers in the Actor and Critic networks. The nonlinearity ELU was used to produce more natural gradients. The implementation also uses parameter space noise instead of the standard UO-process.
