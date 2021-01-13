import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import numpy as np
import pyro
import pyro.contrib.gp as gp
from utilities_full import BayesOpt
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)

def rosenbrock_f(x):
    '''
    Unconstrained Rosenbrock function (objective)
    '''
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_g1(x):
    '''
    Rosenbrock cubic constraint
    g(x) <= 0
    '''
    return (x[0] - 1)**3 - x[1] + 1


def rosenbrock_g2(x):
    '''
    Rosenbrock linear constraint
    g(x) <= 0
    '''
    return x[0] + x[1] - 1.8


f1 = rosenbrock_f
g1 = rosenbrock_g1
g2 = rosenbrock_g2


bounds = np.array([[-1.5,1.5],[-0.5,0.5]])
x0 = [0.5,0.5]
solution1 = BayesOpt().solve(f1, x0, bounds=bounds.T, print_iteration=True, constraints=[g1,g2])
