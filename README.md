# LAPAC: Latent Autonomous Planning Actor Critic

The LAPAC algorithm was developed as a course project for [COMP
767](https://www.cs.mcgill.ca/~siamak/COMP767/index.html) --
Probabilistic Graphical Models -- at McGill University, given by Professor
[Siamak Ravanbakhsh](https://www.cs.mcgill.ca/~siamak/) in the Fall 2019 semester.

LAPAC is a maximum entropy reinforcement learning algorithm that is designed
particularly for partially-observable environments. It uses a fully stochastic
latent variable model inspired by [SLAC](https://github.com/alexlee-gk/slac),
and uses the generative latent dynamics model to train the actor and critic
offline.
