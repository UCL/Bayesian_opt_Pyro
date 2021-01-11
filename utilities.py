import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to
import sobol_seq
import pyDOE
import pyro
import pyro.contrib.gp as gp
import numpy  as np
np.random.seed(0)
torch.seed()

class BayesOpt(object):
    def __init__(self):
        print('Start Bayesian Optimization using Torch')


    def solve(self, objective, xo=0.5, bounds=(0,1), maxfun=20, N_initial=4, select_kernel='Matern52'):

        self.x0        = torch.Tensor(xo)
        self.bounds    = bounds
        self.maxfun    = maxfun
        self.N_initial = N_initial
        self.objective = objective
        self.kernel    = select_kernel
        self.nx = max(self.x0.shape)

        self.X = torch.from_numpy(pyDOE.lhs(self.nx, N_initial))


        sol  = self.run_main()
        return sol

    def run_initial(self):
        Y = torch.zeros([self.N_initial])
        for i in range(self.N_initial):
            Y[i] = self.compute_function(self.X[i,:], self.objective).reshape(-1,)
        return Y

    def training(self):

        Y_unscaled  = self.Y
        X_unscaled   = self.X
        nx = self.nx

        # Scale the variables
        self.X_mean, self.X_std = X_unscaled.mean(axis=0), X_unscaled.std( axis=0)

        self.Y_mean, self.Y_std = Y_unscaled.mean( axis=0), Y_unscaled.std( axis=0)
        self.X_norm, self.Y_norm = (X_unscaled - self.X_mean) / self.X_std,\
                                   (Y_unscaled - self.Y_mean) / self.Y_std

        X, Y = self.X_norm, self.Y_norm
        # pyro.clear_param_store()

        self.gpmodel.set_data(X, Y)
        # optimize the GP hyperparameters using Adam with lr=0.001
        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
        gp.util.train(self.gpmodel, optimizer)


        return self.gpmodel

    def acquisition_func(self, X_unscaled, acquisition=1):
        X_unscaled = X_unscaled.reshape((1,-1))
        x = (X_unscaled - self.X_mean) / self.X_std

        if acquisition==0:
            mu, _ = self.gpmodel(x, full_cov=False, noiseless=False)
            ac = mu
        elif acquisition==1:
            mu, variance = self.gpmodel(x, full_cov=False, noiseless=False)
            sigma = variance.sqrt()
            ac = mu - 2 * sigma
        elif acquisition==2:
            print(NotImplementedError)
            ac = 0
        else:
            print(NotImplementedError)
            ac = 0
        return ac

    def find_a_candidate(self, x_init):
        # transform x to an unconstrained domain
        constraint = constraints.interval(torch.from_numpy(self.bounds[0]).type(torch.FloatTensor),
                                          torch.from_numpy(self.bounds[1]).type(torch.FloatTensor))
        unconstrained_x_init = transform_to(constraint).inv(x_init)
        unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
        minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')

        def closure():
            minimizer.zero_grad()
            x = transform_to(constraint)(unconstrained_x)
            y = self.acquisition_func(x)
            autograd.backward(unconstrained_x, autograd.grad(y, unconstrained_x))
            return y

        minimizer.step(closure)
        # after finding a candidate in the unconstrained domain,
        # convert it back to original domain.
        x = transform_to(constraint)(unconstrained_x)
        return x.detach()

    def next_x(self, num_candidates=10):
        candidates = []
        values = []

        #x_init = self.gpmodel.X[-1:]
        x_init = torch.from_numpy(self.generate_samples_for_multistart(num_candidates)).type(torch.FloatTensor)
        for i in range(num_candidates):
            x = self.find_a_candidate(x_init[i])
            y = self.acquisition_func(x)
            candidates.append(x)
            values.append(y)
        # x_plot = torch.Tensor(np.linspace(0, 1, 100))
        # y_plot = torch.Tensor(np.linspace(0, 1, 100))
        #
        # for i in range(100):
        #     y_plot[i] = self.acquisition_func(x_plot[[i]])
        # plt.plot(x_plot,y_plot.detach().numpy())
        # print('--')
        # print(self.gpmodel.kernel.lengthscale_unconstrained)
        # print(self.gpmodel.kernel.variance_unconstrained)
        argmin = torch.min(torch.cat(values), dim=0)[1].item()
        return candidates[argmin]

    def generate_samples_for_multistart(self, multi_start=5):
        multi_startvec = sobol_seq.i4_sobol_generate(self.nx, multi_start)

        multi_startvec_scaled = multi_startvec *(self.bounds[1]-self.bounds[0]) + self.bounds[0]
        return  multi_startvec_scaled

    def update_data(self, xmin):
        xmin = xmin.reshape(-1,)
        y = self.compute_function(xmin, self.objective).reshape(-1,)
        self.X = torch.cat([self.X,xmin.reshape(1,self.nx)])
        self.Y = torch.cat([self.Y,y])


    def run_main(self):
        self.Y = self.run_initial()
        self.define_GP()

        self.gpmodel = self.training()
        for i in range(self.maxfun):
            xmin = self.next_x()

            self.update_data(xmin)
            self.gpmodel = self.training()


        optim = np.argmin(self.Y)
        x_opt = self.X[[optim]]
        y_opt = self.Y[[optim]]

        print('Optimum Objective Found: ', y_opt.numpy())
        print('Optimum point Found: ', x_opt.numpy())

        return solutions( x_opt, y_opt, self.maxfun)

    def define_GP(self):
        X, Y, nx = self.X, self.Y, self.nx
        if self.kernel=='Matern52':
            self.gpmodel = gp.models.GPRegression(X, Y, gp.kernels.Matern52(input_dim=nx,
                                             lengthscale=torch.ones(nx)),
                                             noise=torch.tensor(0.1), jitter=1.0e-4,)
        elif self.kernel=='Matern52':
            self.gpmodel = gp.models.GPRegression(X, Y, gp.kernels.Matern32(input_dim=nx,
                                             lengthscale=torch.ones(nx)),
                                             noise=torch.tensor(0.1), jitter=1.0e-4,)
        elif self.kernel=='RBF':
            self.gpmodel = gp.models.GPRegression(X, Y, gp.kernels.RBF(input_dim=nx,
                                             lengthscale=torch.ones(nx)),
                                             noise=torch.tensor(0.1), jitter=1.0e-4,)
        else:
            print('NOT IMPLEMENTED KERNEL, USE RBF INSTEAD')
            self.gpmodel = gp.models.GPRegression(X, Y, gp.kernels.RBF(input_dim=nx,
                                             lengthscale=torch.ones(nx)),
                                             noise=torch.tensor(0.1), jitter=1.0e-4,)

    def compute_function(self,x_torch, f):
        x = x_torch.detach().numpy().reshape(-1,)
        y = f(x).reshape(-1,)

        return torch.from_numpy(y).type(torch.FloatTensor)
class solutions:
    def __init__(self, x_opt, y_opt, maxfun):
        self.x = x_opt.detach().numpy()
        self.f = y_opt.detach().numpy()
        self.maxfun = maxfun
        self.success = 0

    def __str__(self):
                  '\n Solution xmin = '+ str(self.x)+ \
                  '\n Objective value f(xmin) = '+ str(self.f)+ \
                  '\n With ' + str(self.maxfun) + ' Evaluations '
        return  message
