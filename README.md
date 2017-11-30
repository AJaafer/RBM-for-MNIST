# Bernoulli Restricted Boltzmann Machine (RBM) for MNIST reconstruction

TensorFlow implementation of a Bernoulli Restricted Boltzmann Machine (RBM) for MNIST digits reconstruction.

Credits to [@patricieni]

## Requirements
- Python 2.7 or 3.5
- [TensorFlow 1.0.1+](https://www.tensorflow.org/install/)

## RBM Graphical Model

Restricted Boltzmann Machines are a class of undirected probabilistic graphical models where the nodes in the graph are random variables.

The latter are well known and extensively studied in the physics literature, with the ferromagnetic Ising spin model from statistical mechanics being the best example. Atoms are fixed on a 2-D (or 1-D) lattice and neighbours interact with each other. We consider the energy associated with a spin state of +/- 1 and we are interested in the possible states the system takes. It turns out that the joint probability of such a system is modelled by the Boltzmann (Gibbs) distribution.

Similarly, the joint probability of a restricted boltzmann machine can be modelled by the gibbs distribution. Furthermore, an RBM can be considered a stochastic neural network where you have a set of visible nodes that take some data as input and a set of hidden nodes that encode a lower dimensional representation of that data. Because you can think of your input as a high dimension probability distribution, the goal is to learn the joint probability of the ensemble (visible-hidden).

The model is defined in the rbm.py file, together with methods for computing the probabilities and free energy of the system as well as sampling. The goal is to learn the joint probability distribution that maximizes the probability over the data, also known as likelihood.

- Add joint probability distribution

- Energy (x,h)

- Free Enery (x)

- Derivation

## Inference

- Conditional distribution (h|v)and (v|h)

## Learning

Use the RBM for learning a lower dimensional representation of the MNIST dataset. You can see the reconstructions in both cases and how it's slightly better in the gaussian scenario.



## Usage

Run the main.ipynb file in jupyter

## Results

Under progress

## Extensions

Deep Boltzmann Machines and Deep Belief Networks.

Gaussian RBM

Contrastive Divergence k (for k>1 step of MCMC simulation) w/ weight cost or temperature [Tieleman 08]. 

...
