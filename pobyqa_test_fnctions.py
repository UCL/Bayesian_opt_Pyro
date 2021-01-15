import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import numpy as np
import pyro
import pyro.contrib.gp as gp
from utilities_full import BayesOpt, PenaltyFunctions
assert pyro.__version__.startswith('1.5.1')
pyro.enable_validation(True)  # can help with debugging
pyro.set_rng_seed(1)
import pybobyqa
import functools


import logging
#logging.basicConfig(level=logging.INFO, format='%(message)s')
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

def penalized_objective(f, g, mu, x):
    '''
    :param x: decision variables
    :type x:  numpy array
    :param f: Objective
    :type f:  python function
    :param g: constraints
    :type g: list with python functions
    :return: objective
    :rtype:  numpy array
    '''
    obj   = f(x)
    n_con = len(g)
    for i in range(n_con):
        obj += mu*max(g[i](x),0)**2
    return obj


f1 = rosenbrock_f
g1 = rosenbrock_g1
g2 = rosenbrock_g2


aug_functions = PenaltyFunctions(f1,[g1,g2],type_penalty='le', mu=100)#functools.partial(penalized_objective,f1,[g1,g2], 100)

f_pen = aug_functions.aug_obj

bounds = np.array([[-1.5,1.5],[-1.5,1.5]])
x0 = np.array([0.5,0.5])
user_params = {'logging.save_diagnostic_info': True}
user_params['logging.save_xk'] = True
user_params['logging.save_xk'] = True

soln = pybobyqa.solve(f_pen, x0, bounds=bounds.T, user_params=user_params, maxfun=100)


def quadratic_g(x):
    '''
    test constraint
    g(x) <= 0
    '''
    return 1 - x[0] - x[1]

def quadratic_f(x):
    '''
    test objective
    '''
    return x[0]**2 + 10 * x[1]**2 + x[0] * x[1]
f1 = quadratic_f
g1 = quadratic_g

f_pen = functools.partial(penalized_objective,f1,[g1,g2], 100)

soln1 = pybobyqa.solve(f_pen, x0, bounds=bounds.T)

