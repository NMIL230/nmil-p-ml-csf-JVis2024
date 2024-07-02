'''
This script defines a Gaussian process (GP) classification model and associated utility functions.
It first imports required libraries, including PyTorch, GPyTorch, NumPy, and Matplotlib.
It then defines a custom Bernoulli likelihood class for use in the GP, along with two other classes to perform feature scaling.

Next, the script defines a function for preparing input data for the GP classification model, including scaling and converting
the labels to the range [0,1]. It then defines the GP classification model itself, along with functions for training and
evaluating it. Additionally, a function is provided for creating an evaluation grid for the model and another for plotting
the model's classification results.

Finally, the script defines a function for finding the input values that maximize the differential entropy of the GP's output.
This is a measure of the "uncertainty" of the model's predictions, with higher entropy indicating greater uncertainty.

It includes the following classes and functions:

CustomBernoulliLikelihood: a custom likelihood class used for the GP, which is based on the Bernoulli distribution and includes two hyperparameters: psi_gamma (g) and psi_lambda (l)

logFreq and logContrast: two classes used to transform the input data (frequency and contrast) logarithmically

prepare_data: a function that preprocesses the input data, including separating the features and labels, applying logarithmic transformations, scaling the features, and converting the labels to values between 0 and 1 if they are -1 and 1

GPClassificationModel: a class that represents the GP classification model with a constant mean and a kernel consisting of a linear kernel and an RBF kernel

train_gp_classification_model: a function that trains the GP model using the provided training inputs, labels, and likelihood using the Adam optimizer and the marginal log likelihood as the loss function

plot_classification_results: a function that plots the classification results of the GP model, including the evaluation grid, the predicted probabilities, and the original data points

find_best_entropy: a function that finds the input values that maximize the differential entropy of the GP's output, which is a measure of the "uncertainty" of the model's predictions, with higher entropy indicating greater uncertainty.

random_samples_from_data: Takes in data, and some other params to label the data, and chooses n points from this dataset

simulate_labeling: Labels points using a cubic spline, and psi_sigma, psi_gamma, psi_lambda parameters. See function for detailed comments

create_cubic_spline: Pass in a curve, get a cubic spline

get_data_bounds: returns the min and max for x and y axis

create_evaluation_grid: takes in min and max values for both axis, as well as the size of the grid, and returns a grid

evaluate_posterior_mean: evaluates the GP model (the mean) on some data points

transform_dataset: transforms the passed in dataset (as I'm writing this, it performs the identity transformation (so nothing))
'''
import torch
import gpytorch
from IPython.core.display_functions import clear_output
import warnings
import numpy as np
from gpytorch.means import Mean, ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy, LMCVariationalStrategy
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.likelihoods import _OneDimensionalLikelihood
from gpytorch.functions import inv_matmul, log_normal_cdf
from torch.distributions import Normal, Bernoulli
import torch.distributions as dist    
from torch.nn import functional as F
from scipy.interpolate import CubicSpline
from scipy.special import erf
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import qmc
from pathlib import Path
from PIL import Image
import os
import time
import json
import copy
from math import comb
import itertools
from itertools import combinations_with_replacement


class CustomBernoulliLikelihood(_OneDimensionalLikelihood):
    """
    Bernoulli likelihood object with two hyperparameters used to fit the GP model to the training data
    """

    def __init__(self, g, l):
        """
        :param g: psi_gamma - the percentage of guesses you get right when random guessing
        :param l: psi_lambda - the percetnage of guesses you get wrong when you know what you're doing
        """
        super().__init__()
        if not (0 <= g <= 1):
            raise ValueError("g must be in [0, 1]")
        if not (0 <= l <= 1):
            raise ValueError("l must be in [0, 1]")
        self.g = torch.tensor(g)
        self.l = torch.tensor(l)
        self.quadrature = GaussHermiteQuadrature1D()

    def forward(self, function_samples, **kwargs):
        output_probs = Normal(0, 1).cdf(function_samples)
        output_probs = output_probs.mul(1 - self.l - self.g).add(self.g)
        return Bernoulli(probs=output_probs)

    def marginal(self, function_dist, **kwargs):
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = Normal(0, 1).cdf(link)
        output_probs = output_probs.mul(1 - self.l - self.g).add(self.g)
        return Bernoulli(probs=output_probs)
    
    # default _OneDimensionalLikelihood expected_log_prob used


class CustomMeanModule(Mean):
    def __init__(self, models_and_likelihoods, scale_factor=1.0, gaussian_lengthscale=None):
        '''
        models_and_likelihoods: a list of tuples of this form - [(model1, likelihood1), (model2,likelihood2),...]
        '''
        super().__init__()
        self.models_and_likelihoods = models_and_likelihoods
        self.scale_factor = scale_factor
        self.gaussian_lengthscale = gaussian_lengthscale

    def forward(self, x):

        # Compute the latent distribution
        total_mean = 0
        n = 0
        with torch.no_grad():
            for gp_model, likelihood in self.models_and_likelihoods:
                n += 1
                gp_model.eval()
                likelihood.eval()
                latent_pred = gp_model(x)
                # posterior_pred = self.likelihood(latent_pred)
                total_mean += latent_pred.mean
        # Return the scaled posterior mean
        avg = (total_mean / n)
        if self.gaussian_lengthscale is None:
            return self.scale_factor * avg
        else:
            return self.scale_factor * (avg - (
                    avg * np.exp(-(1 / (self.gaussian_lengthscale * self.gaussian_lengthscale)) * avg * avg)))


class logFreq:
    """
    A class that defines a logarithmic frequency transformation.
    """

    def __init__(self):
        self.n = 0.125  # the smallest raw freq we expect

    def forward(self, data):
        return np.log2(data / self.n)

    def inverse(self, transformed_data):
        return self.n * np.power(2.0, transformed_data)


class logContrast:
    """
    A class that defines a invert-logarithmic contrast transformation.
    """

    def __init__(self):
        self.n = 1  # the smallest contrast sensitivity we expect (1/contrast is in the range [1, inf)

    def forward(self, data):
        return -1 * np.log10(data * self.n)  # the '-' inverts the data!!!

    def inverse(self, transformed_data):
        return self.n * np.power(10.0, -1 * transformed_data)


def prepare_data(data, raw=True, neg_labels=False):
    """
    A function that preprocesses the input data, which includes separating the features and labels, applying logarithmic
    transformations, scaling the features into the specified range, and converting the labels to values between 0 and 1
    if neg_labels is True.
    :param data: nx(2+1) matrix. n data points, 2 dimensions, and the 3rd column is the label
    :param range: We will scale each dimension to be within this range
    :param neg_labels: Signifies the labels are -1 and 1. If so, we convert them to 0 and 1
    """

    # Separate the features and labels
    X = data[:, :2].copy()
    y = data[:, 2].copy()

    if raw:
        freq_processor = logFreq()
        X[:, 0] = freq_processor.forward(X[:, 0])
        contrast_processor = logContrast()
        X[:, 1] = contrast_processor.forward(X[:, 1])

    # Convert the labels to values between 0 and 1 if neg_labels is True
    if neg_labels:
        # y = (y + 1) / 2   # there could be some rounding issues using this method
        y[y == -1] = 0
        warnings.warn("Labels should not be -1; please use 0 instead.", DeprecationWarning)

    # Convert the data to PyTorch tensors
    Xt = torch.from_numpy(X).float()
    yt = torch.from_numpy(y).float()

    return Xt, yt, X, y

class MixtureNormalPrior(gpytorch.priors.Prior):    
    '''
    Prior enforced using two normal distributions with a specified weight for each distribution
    '''
    def __init__(self, mean1, std1, mean2, std2, weight1):    
        super().__init__(validate_args=False)    
        self.mean1 = mean1    
        self.std1 = std1    
        self.mean2 = mean2    
        self.std2 = std2    
        self.weight1 = weight1    
    def log_prob(self, x):
        prob1 = self.weight1 * torch.distributions.Normal(self.mean1, self.std1).log_prob(x)    
        prob2 = (1 - self.weight1) * torch.distributions.Normal(self.mean2, self.std2).log_prob(x)    
        return torch.logsumexp(torch.stack([prob1, prob2]), dim=0)

class GPClassificationModel(ApproximateGP):    

    """    
    A class that represents a Gaussian process classification model with a constant mean and a kernel consisting of a linear kernel and a RBF kernel.    
    """    

    def __init__(self, train_x, mean_module, kernel_config='new', min_lengthscale=None):    
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))    
        variational_strategy = VariationalStrategy(    
            self, train_x, variational_distribution, learn_inducing_locations=False    
        )    
        super(GPClassificationModel, self).__init__(variational_strategy)    
        self.mean_module = mean_module
            
        # Linear kernel with constraints and priors    
        self.linear_kernel = LinearKernel(active_dims=[1])
        linear_variance_constraint = GreaterThan(.1)    
        self.linear_kernel.variance_constraint = linear_variance_constraint    

        # RBF kernel with constraints and priors    
        if min_lengthscale is not None:    
            RBF_lengthscale_constraint = GreaterThan(min_lengthscale)

        if kernel_config == 'new':
            self.rbf_kernel = RBFKernel(ard_num_dims=2,lengthscale_constraint=RBF_lengthscale_constraint)
        elif kernel_config == 'old':
            self.rbf_kernel = RBFKernel(active_dims=[0],lengthscale_constraint=RBF_lengthscale_constraint)

        # Set constraints and priors for the outputscale of rbf_kernel    
        rbf_outputscale_constraint = GreaterThan(.1)    
        self.rbf_kernel.outputscale_constraint = rbf_outputscale_constraint    

        # Combining the kernels    
        self.covar_module = ScaleKernel(self.linear_kernel + self.rbf_kernel)

    def forward(self, x):    
        mean_x = self.mean_module(x)    
        covar_x = self.covar_module(x)    
        latent_pred = MultivariateNormal(mean_x, covar_x)    
        return latent_pred
    
class MTGPClassificationModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        num_latents=2,
        num_tasks=2,
        min_lengthscale=0.15,
        kernel_config='new',
        lmc_reg=False,
        lmc_init_type='identity',
        grad_scale_factor=1e-3,
        grad_mask_type='off_diagonal'
        ):

        inducing_points = torch.rand(num_latents, 50, 2)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(-2),
            batch_shape=torch.Size([num_latents])
        )
        variational_strategy = LMCVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1
        )
        super().__init__(variational_strategy)
        
        # Minimum value for variance and outputscale
        min_hyperparam = .1

        # Mean module
        self.mean_module = ConstantMean(batch_shape=torch.Size([num_latents]))
        
        # Linear kernel with variance constraint
        self.linear_kernel = LinearKernel(active_dims=[1], batch_shape=torch.Size([num_latents]))
        self.linear_kernel.register_constraint("raw_variance", GreaterThan(min_hyperparam))

        # RBF kernel with constraints
        if min_lengthscale is not None:
            RBF_lengthscale_constraint = GreaterThan(min_lengthscale)
        else:
            RBF_lengthscale_constraint = None

        if kernel_config == 'new':
            self.rbf_kernel = RBFKernel(ard_num_dims=2,lengthscale_constraint=RBF_lengthscale_constraint)
        elif kernel_config == 'old':
            self.rbf_kernel = RBFKernel(active_dims=[0],lengthscale_constraint=RBF_lengthscale_constraint)
        
        # Set constraints and priors for the outputscale of rbf_kernel    
        rbf_outputscale_constraint = GreaterThan(.1)    
        self.rbf_kernel.outputscale_constraint = rbf_outputscale_constraint    

        # Combining the kernels
        self.covar_module = ScaleKernel(self.linear_kernel + self.rbf_kernel)
        
        if lmc_reg:
            # Initialize LMC coefficients dynamically
            if lmc_init_type == 'identity':
                lmc_coefficients = torch.eye(num_tasks, num_latents)
            elif lmc_init_type == 'ones':
                lmc_coefficients = torch.ones(num_tasks, num_latents)
            
            self.variational_strategy.register_parameter("lmc_coefficients", torch.nn.Parameter(lmc_coefficients))
            
            # Initialize the gradient mask based on grad_mask_type
            self.lmc_mask = torch.full((num_tasks, num_latents), grad_scale_factor)
            if grad_mask_type == 'off_diagonal':
                self.lmc_mask.fill_diagonal_(1.0)
            elif grad_mask_type == 'diagonal':
                self.lmc_mask.fill_diagonal_(grad_scale_factor)
                self.lmc_mask = 1.0 - self.lmc_mask + self.lmc_mask.fill_diagonal_(grad_scale_factor)

            # Register the backward hook
            self.variational_strategy.lmc_coefficients.register_hook(self.apply_lmc_mask)
        
        self.grad_scale_factor = grad_scale_factor

    def apply_lmc_mask(self, grad):
        """
        Apply the mask to the gradient. Elements in the gradient are either scaled 
        or left unchanged based on the lmc_mask.
        """
        scaled_grad = grad * self.lmc_mask

        return scaled_grad
    
    def forward(self, Xt):
        mean_x = self.mean_module(Xt)
        covar_x = self.covar_module(Xt)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_classification_model(model, likelihood, Xt, yt, beta=1, training_iterations=2000, lr=0.1,
                                  visualize_loss=False, progress_bar=True, task_indices=None, multitask_flag=False,
                                  weight_decay=0, patience=20, min_delta=0.005):
    """
    A function that trains the GP model using the provided training inputs, labels, and likelihood. It uses the Adam
    optimizer and the marginal log likelihood as the loss function.
    :param model: GP model
    :param likelihood: Likelihood (i.e. bernoulli, or custom bernoulli)
    :param Xt: Training data (shape?)
    :param yt: Labels (shape?)
    :param training_iterations: how many iterations you want to optimize the hyperparams
    :param lr: learning rate
    :param beta: Regularizer. Smaller beta is more regularized.
    :param visualize_loss: Debugging mostly. Plots loss inline. Default False.
    """
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    if task_indices is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.parameters(), 'weight_decay': weight_decay},  # Apply weight decay here
            {'params': likelihood.parameters()}  # No weight decay for likelihood parameters
        ], lr=lr)

        
    # "Loss" for GPs - the marginal log likelihood
    # num_data refers to the number of training datapoints
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, yt.numel(), beta=beta, combine_terms=False)

    # Initialize loss history
    loss_history = []

    iterator = range(training_iterations)
    if progress_bar:
        iterator = tqdm(range(training_iterations), desc="Training progress", ncols=100)

    # Early stopping parameters
    best_loss = float('inf')
    num_bad_epochs = 0

    for i in iterator:
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        
        if task_indices is None:
            output = model(Xt)
        else:
            output = model(Xt, task_indices=task_indices)
        
        log_lik, kl_div, log_prior = mll(output, yt) # DM: this is how we did it for MLA Gpytrch no idea why I don't care tho
        loss = -(log_lik - kl_div + log_prior) # DM: this is how we did it for MLA Gpytrch no idea why I don't care tho

        # loss = -mll(output, yt)
        current_loss = loss.item()
        if visualize_loss: loss_history.append(current_loss)

        # Early stopping logic (skip the first iteration i > 0)
        if i > 0:
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                num_bad_epochs = 0
            else:
                num_bad_epochs += 1

            if num_bad_epochs >= patience:
                # print(f"Early stopping triggered at iteration {i+1}")
                break

        loss.backward()
        optimizer.step()

        # Update progress bar and plot the loss if visualize_loss is True
        if visualize_loss and (i + 1) % 10 == 0:
            clear_output(wait=True)
            plt.plot(loss_history, label="Loss")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

def getQcsfRMSE(xx, cs, peakSensitivity, peakFrequency, logBandwidth, delta, qcsf, get_grid_predictions=False):
    """
    Args:
        xx (numpy.ndarray): Mesh grid X coordinate.
        cs (scipy.interpolate.CubicSpline object): ground truth CSF.
        peakSensitivity (float): sensitivity greatest value on truncated log-parabola
        over all frequencies
        peakFrequency (float): log2(CPD) frequency value of peakSensitivity
        logBandwidth (float): log2(CPD) full width at half max
        delta (float): sensitivity difference between truncated line and peakSensitivity
        qcsf (function): qcsf predicting function

    Returns:
        Root Mean Square Error (RMSE) calculated on each for each well-defined frequency value
        on our mesh grid. Returns in units of log_10(contast sensitivity).

    """
    xmedium = xx[0, :]

    # Get Frequency defined on same values as xmedium
    frequency = logFreq().inverse(xmedium)

    # Get csf Predictions
    ymedium = qcsf(peakSensitivity, peakFrequency, logBandwidth, delta, frequency).reshape(-1)

    # Calculate Ground truth posterior using spline.
    discrete_x2 = cs(xmedium)

    # Calculate RMSE between qCSF prediction, ground truth where ground truth well defined (>0).
    rmse = np.sqrt(np.mean((ymedium[discrete_x2 > 0] - discrete_x2[discrete_x2 > 0]) ** 2))
    
    if get_grid_predictions:
        return rmse, ymedium
    
    return rmse


def getRMSE(xx, yy, zz, level, cs, error='RMSE', orientation='vertical'):
    """
    Args:
        xx (numpy.ndarray): Mesh grid X coordinate.
        yy (numpy.ndarray): Mesh grid Y coordinate.
        zz (numpy.ndarray): evaluate_posterior_mean(model, likelihood, grid_transformed) reshaped to xx.shape
        level: inflection point on psychometric curve (point of interest) (1-psi_lambda+psi_gamma)/2.
        cs: scipy.interpolate.CubicSpline object of ground truth CSF.
        orientation (str): 'horizontal' or 'vertical' - determines the direction of RMSE calculation.
    """
    zzmin = (zz - level) ** 2

    if orientation == 'vertical':
        # Original method for vertical RMSE
        yindex = np.int64(np.argmin(zzmin, axis=0))
        yplot = yy[:, 0]
        xmedium = xx[0, :]
        ymedium = yplot[yindex[:]]
        discrete_x2 = cs(xmedium)

    elif orientation == 'horizontal':
        # New method for horizontal RMSE
        xindex = np.int64(np.argmin(zzmin, axis=1))
        xplot = xx[0, :]
        ymedium = yy[:, 0]
        xmedium = xplot[xindex[:]]
        discrete_y2 = cs(ymedium)

    else:
        raise ValueError("Orientation must be either 'horizontal' or 'vertical'")

    if error == 'RMSE':
        if orientation == 'vertical':
            rmse = np.sqrt(np.mean((ymedium[discrete_x2 > 0] - discrete_x2[discrete_x2 > 0]) ** 2))
            error = rmse

    elif error == 'RSE':
        if orientation == 'vertical':
            rse = np.sqrt((ymedium[discrete_x2 > 0] - discrete_x2[discrete_x2 > 0]) ** 2)
            error = rse
        elif orientation == 'horizontal':
            rse = (xmedium[discrete_y2 > 0] - discrete_y2[discrete_y2 > 0])
            error = rse

    return error


def plot_classification_grid(X_eval, xx, yy, Z_gpy, X, y, phi, res=11, figsize=(10, 5), x_range=None, y_range=None):
    """
    A function that plots the classification results of the GP model, including the evaluation grid, the predicted probabilities, and the original data points.

    Args:
        X_eval (torch.Tensor): Evaluation grid points as a tensor.
        xx (numpy.ndarray): Mesh grid X coordinate.
        yy (numpy.ndarray): Mesh grid Y coordinate.
        Z_gpy (numpy.ndarray): Predicted probability of being in class 1.
        X (numpy.ndarray): Unscaled input features.
        y (numpy.ndarray): Input labels.
        phi (float): Threshold probability for classification boundary.
        res (int): Number of levels for the contourf plot.
        figsize (tuple): Figure size.
        x_range (tuple): Range limits for the x-axis.
        y_range (tuple): Range limits for the y-axis.
    """
    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    f.set_facecolor('white')
    cs = plt.contour(xx, yy, Z_gpy, levels=[phi], colors='black', linestyles='dashed', linewidths=2)
    plt.clabel(cs, fontsize=20)

    plt.contourf(xx, yy, Z_gpy, levels=np.linspace(0, 1, res))

    plt.colorbar()

    plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Seen', marker='+', c='blue', s=100)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Unseen', marker='d', edgecolors='red', facecolors='none', s=100)
    plt.title('CSF')
    plt.xlabel("Spatial Frequency (Log2 cpd)")
    plt.ylabel("-Log10(Contrast)")
    plt.legend()

    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    plt.plot()
    plt.show()

def find_best_entropy(model, likelihood, X_eval, xx, Xt, task_index=None, acq_config = 'new', multitask_flag=False):
    """
    A function that computes the differential entropy on the evaluation grid, finds the indices of the maximum
    differential entropy, and computes the corresponding coordinates.
    """
    # Helper function for entropy
    H = lambda x: -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)
    C = torch.sqrt(3.1415 * torch.log(torch.tensor(2.)) * 0.5)

    # Ensure no gradients are computed
    with torch.no_grad():  
        # get mean and variance of evaluation grid
        output = likelihood(model(X_eval))
        mean_eval = output.mean
        variance_eval = output.variance
        
        # get only a specific task if conjoint
        if task_index is not None:
            mean_eval = mean_eval[:,task_index]
            variance_eval = variance_eval[:,task_index]

    # Compute standard deviation once
    sd = variance_eval.sqrt()

    # Compute differential entropy components
    Denom1 = (sd ** 2 + 1).sqrt_()
    first_pre = torch.distributions.Normal(0., 1.).cdf(mean_eval / Denom1)
    first = H(first_pre)
    Nom = torch.ones_like(mean_eval) * C
    Denom2 = sd ** 2 + Nom ** 2
    second = (Nom / Denom2.sqrt_()) * torch.exp(-mean_eval ** 2 / (2 * Denom2))
   
    # Compute differential entropy
    de = first - second
    
    # Find indices of maximum differential entropy
    sorted_indices = torch.argsort(de, descending=True)
    
    # Return the entropies grid (so it can be plotted if you want)
    entropies = de.reshape(xx.shape)

    # acq_config necessary but nearest neighbors needs at least 4 real points for whatever reason
    # (and we don't calculate distance from ghost points)

    if acq_config == 'new' and len(Xt) >= 12:
    
        # Find indices of the top 5% of entropies
        num_top_entropies = int(len(de) * 0.025)
        top_entropy_indices = sorted_indices[:num_top_entropies]

        # Convert to NumPy array if necessary
        X_eval_np = X_eval.cpu().detach().numpy() if isinstance(X_eval, torch.Tensor) else X_eval

        # Fit nearest neighbors on the labeled data (Xt)
        nn = NearestNeighbors(n_neighbors=1)

        # skip the eight ghost points
        nn.fit(Xt[8:])

        # Find the distance to the nearest labeled point for each point in the top 5% entropies
        top_entropy_points = X_eval_np[top_entropy_indices]
        distances, _ = nn.kneighbors(top_entropy_points)
        
        # Select the index that is furthest from its neighbors
        furthest_point_index = top_entropy_indices[np.argmax(distances)]

        # Replace the first index in sorted_indices with the furthest_point_index
        sorted_indices[0] = furthest_point_index

    return entropies, sorted_indices, mean_eval, variance_eval


def plot_entropy_grid(X_eval, xx, yy, entropies, best_x, best_y, phi, res=11, figsize=(10, 5), x_range=None,
                      y_range=None):
    """
    A function that plots the classification results of the GP model, including the evaluation grid, the predicted probabilities, and the original data points.
    Args:
        X_eval (torch.Tensor): Evaluation grid points as a tensor.
        xx (numpy.ndarray): Mesh grid X coordinate.
        yy (numpy.ndarray): Mesh grid Y coordinate.
        entropies (torch): Entropies within grid
        X (numpy.ndarray): Unscaled input features.
        y (numpy.ndarray): Input labels.
        res (int): Number of levels for the contourf plot.
        figsize (tuple): Figure size.
        x_range (tuple): Range limits for the x-axis.
        y_range (tuple): Range limits for the y-axis.
    """
    # Initialize fig and axes for plot
    f, ax = plt.subplots(1, 1, figsize=figsize)
    f.set_facecolor('white')
    plt.contourf(xx, yy, entropies, levels=np.linspace(torch.min(entropies), torch.max(entropies), res))

    plt.colorbar()

    plt.scatter(best_x, best_y, label='Max', marker='+', c='black', s=100)
    plt.title('Entropy')
    plt.xlabel("Spatial Frequency (Log2 cpd)")
    plt.ylabel("-Log10(Contrast)")
    plt.legend()

    if x_range is not None:
        plt.xlim(x_range)
    if y_range is not None:
        plt.ylim(y_range)

    plt.plot()
    plt.show()


def random_samples_from_data(data, cs, psi_gamma, psi_lambda, n, replacement=False, inds=None, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    """
      Draws n random data points
      Labels produced via cs, psi_sigma, psi_gamma, psi_lambda
      :param data: mx2 matrix - the data to draw points from and label them
      :param cs: a CubicSpline
      :param psi_sigma: spread parameter
      :param sigmoid_type: which shape sigmoid to use
      :param psi_gamma: success percentage when guessing at random
      :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right
      :param n: number of data points to generate
      :param replacement: set to True if you want to sample with replacement
      :param inds: Optional, the specific indices of the grid you want to sample
      :return: X, a nx2 matrix, and y, a length n vector
    """
    m = data.shape[0]

    if replacement is False and m < n:
        n = m
        warnings.warn("You are attempting to sample more points than are in the grid." +
                      "This is likely a mistake", DeprecationWarning)

    valid_indices = np.arange(0, m)
    if inds is None:
        chosen_indices = np.random.choice(valid_indices, size=n, replace=replacement)
    else:
        chosen_indices = inds

    x1 = data[chosen_indices, 0]
    x2 = data[chosen_indices, 1]

    y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)	

    # return the generated values and labels
    X = np.vstack((x1, x2)).T
    return X, y

def get_halton_samples(xx, yy, n):	
    l_bounds = [0, 0]
    u_bounds = [xx.shape[1], xx.shape[0]]

    sampler = qmc.Halton(d=2, scramble=False)

    samples = sampler.integers(l_bounds, u_bounds=u_bounds, n=n)
    samples = np.array(samples)

    x1_indices = samples[:, 0]
    x2_indices = samples[:, 1]

    x1 = xx[0, x1_indices]
    x2 = yy[x2_indices, 0]

    X = np.vstack((x1, x2)).T

    return X

def halton_samples_from_data(xx, yy, cs, psi_gamma, psi_lambda, n, strict=False, sigmoid_type='logistic', psi_sigma=.08):	
    l_bounds = [0, 0]
    u_bounds = [xx.shape[1], xx.shape[0]]

    sampler = qmc.Halton(d=2, scramble=False)

    samples = sampler.integers(l_bounds, u_bounds=u_bounds, n=n)
    samples = np.array(samples)

    x1_indices = samples[:, 0]
    x2_indices = samples[:, 1]

    x1 = xx[0, x1_indices]
    x2 = yy[x2_indices, 0]

    X = np.vstack((x1, x2)).T
    y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)

    return X, y


def predetermined_samples_from_data(xx, yy, cs, psi_gamma, psi_lambda, n, x_resolution, y_resolution, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    # Initialize variables for the loop
    hit_found = False
    attempts = 0

    while not hit_found and attempts < 5:
        # Define the sample position at (0, 0) scaled by resolution
        predetermined_samples = np.array([[0, 0]]) * np.array([x_resolution, y_resolution])

        # Find the closest indices in the xx, yy grids to the sample positions
        x_indices = np.clip(np.round(predetermined_samples[:, 0]).astype(int), 0, xx.shape[1] - 1)
        y_indices = np.clip(np.round(predetermined_samples[:, 1]).astype(int), 0, yy.shape[0] - 1)

        # Extract the x and y coordinates based on the indices
        x1 = xx[0, x_indices]
        x2 = yy[y_indices, 0]

        # Combine x and y coordinates into sample points
        X = np.vstack((x1, x2)).T

        # Simulate labeling
        y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)

        # Check if a hit is found
        if y[0] == 1:
            hit_found = True
        else:
            attempts += 1

    return X, y

def get_points_at_indices(xx, yy, cs, psi_gamma, psi_lambda, indices, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    '''
    Assuming indices is nx2 array
    '''

    indices = np.array(indices)
    x1_indices = indices[:, 0]
    x2_indices = indices[:, 1]

    x1 = xx[0, x1_indices]
    x2 = yy[x2_indices, 0]
    X = np.vstack((x1, x2)).T

    y = simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=strict, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)

    return X, y


def simulate_labeling(x1, x2, cs, psi_gamma, psi_lambda, strict=False, sigmoid_type='logistic', psi_sigma=.08):
    """
    Labels the points according to the specified sigmoid and spline
	:param x1: vector - First features	
    :param x2: vector - Second features	
    :param cs: a CubicSpline	
    :param psi_gamma: success percentage when guessing at random	
    :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right	
    :param strict: if True, labels are deterministically set based on the threshold level	
    :param mode: 'logistic' or 'probit' to choose the function type	
    :return: the labels	
    """

    probabilities = simulate_sigmoid(x1, x2, cs, psi_gamma, psi_lambda, sigmoid_type=sigmoid_type, psi_sigma=psi_sigma)
    random_values = np.random.uniform(size=probabilities.shape[0])
    level = (1 - psi_lambda + psi_gamma) / 2

    if strict:
        y = (level < probabilities).astype(int)
    else:
        y = (random_values < probabilities).astype(int)

    return y


def simulate_sigmoid(x1, x2, cs, psi_gamma, psi_lambda, sigmoid_type='logistic', psi_sigma=.08):	
    """	
    See comment below for details	
    :param x1: vector - First features	
    :param x2: vector - Second features	
    :param cs: a CubicSpline	
    :param psi_gamma: success percentage when guessing at random	
    :param psi_lambda: (lambda) lapse rate, i.e. the error percentage when you should get it right	
    :param sigmoid_type: 'logistic' or 'normal_cdf' to choose the function type	
    :return: the labels	
    """

    '''
      A classic sigmoid produces outputs between 0 and 1. We want values between psi_gamma and 1 - psi_lambda.
      To do this, we shrink everything by a factor of 1 - psi_lambda - psi_gamma, then add psi_gamma at the end. You
      can check this gives us the correct output.
      We use psi_sigma to affect the rate at which moving to the right/left increases/decreases the output of the sigmoid.
      A smaller psi_sigma means the sigmoid is "steeper" or "squished horizontally" --- larger rate of increase per horizontal step.
      If a y value is above the curve, we want to move it to the left on the sigmoid (-1 labels).
      If a y value is below the curve, we want to move it to the right on the sigmoid (+1 labels).
      We do this by inputting true_y - y as the input to the sigmoid. You can check for yourself that
      this gives us the desired properties.
      We use these values as probabilites, and generate random values between 0 and 1, and the odds that a value
      is < a probability is that probability itself. So we just see if it's less than the probability
      '''
    c = 1 - psi_lambda - psi_gamma	
    true_y = cs(x1)	
    warnings.filterwarnings('ignore')	
    	
    if sigmoid_type == 'logistic':	
        probabilities = c / (1 + np.exp(-1.0/psi_sigma * (true_y - x2))) + psi_gamma	
    elif sigmoid_type == 'normal_cdf':	
        probabilities = c * 0.5 * (1 + erf((true_y - x2) / (psi_sigma * np.sqrt(2)))) + psi_gamma	
    else:	
        raise ValueError("Invalid sigmoid_type. Choose either 'logistic' or 'normal_cdf'.")	
    if len(probabilities.shape) == 0:	
        return np.array([probabilities])	
    return np.array(probabilities)


def create_cubic_spline(curve):
    """
  This method creates a cubic spline to approximate the given curve.
  :param curve: A nx2 numpy matrix. First column is x values. Second column is y values.
  :return: The cubic spline.
  """
    x = curve[:, 0]
    y = curve[:, 1]

    cs = CubicSpline(x, y)
    return cs


def get_data_bounds(data):
    """
  This method returns the left, right, upper, lower bounds for a data set
  :param data: A nx2 numpy matrix. First column is x values. Second column is y values.
  :return: The 4 bounds (left, right, bottom, top)
  """

    x_min = np.min(data[:, 0])
    x_max = np.max(data[:, 0])
    y_min = np.min(data[:, 1])
    y_max = np.max(data[:, 1])

    return x_min, x_max, y_min, y_max

    # return the generated values and labels
    return raw_x1, raw_x2, y


def create_evaluation_grid_resolution(x_min, x_max, y_min, y_max, x_resolution=15, y_resolution=30):
    """
    Creates an evaluation grid covering the specified bounds
    :param x_min: min of x axis
    :param x_max: max of x axis
    :param y_min: min of y axis
    :param y_max: max of y axis
    :param x_resolution: Number of grid columns per spatial frequency octave (default 15)
    :param y_resolution: Number of grid rows per contrast decade (default 30)
    :return: The evaluation data, the x grid points (mesh), the y grid points (mesh)
    """
    x_side_length = int((x_max - x_min) * x_resolution + 1)
    y_side_length = int((y_max - y_min) * y_resolution + 1)

    X_eval, xx, yy = create_evaluation_grid(x_min, x_max, y_min, y_max, x_side_length, y_side_length)

    return X_eval, xx, yy, x_side_length, y_side_length


def create_evaluation_grid(x_min, x_max, y_min, y_max, x_side_length, y_side_length):
    """
    Creates an evaluation grid covering the specified bounds
    :param x_min: min of x axis
    :param x_max: max of x axis
    :param y_min: min of y axis
    :param y_max: max of y axis
    :param side_length: If side length is 10, there will be 100 data points in the grid
    :return: The evaluation data, the x grid points (mesh), the y grid points (mesh)
    """

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, x_side_length), np.linspace(y_min, y_max, y_side_length))

    # creates a (side_length^2)x2 matrix -- the test data
    X_eval = np.vstack((xx.reshape(-1), yy.reshape(-1))).T

    return X_eval, xx, yy

def evaluate_posterior_mean(model, likelihood, X, mean_eval=None, variance_eval=None, task_index=None, multitask_flag = False):
    """ 
      Evaluates the GP model on X
      :param model: GP model
      :param likelihood: likelihood (bernoulli)
      :param X: nx2 matrix, the data
      :return: the posterior mean evaluated at X
      """

    model.eval()
    likelihood.eval()

    if mean_eval is None or variance_eval is None:
        # Perform the forward pass if predictions are not provided
        with torch.no_grad():
            observed_pred = likelihood(model(X))
            mean_eval = observed_pred.mean
            variance_eval = observed_pred.variance

            if task_index is not None:
                mean_eval = mean_eval[:,task_index]
                variance_eval = variance_eval[:,task_index]

    predicted_values = mean_eval.numpy()
    return predicted_values

def transform_dataset(data, phi=None):
    """
   Transforms the data (logged data) to be used by the GP.
   :param data: nx2 matrix - your data
   :param phi: the function to transform your data
   :return: the transformed data
   """

    # don't modify the original data
    X = data.copy()

    if phi is not None:
        X = phi(X)

    # Convert the data to PyTorch tensors
    Xt = torch.from_numpy(X).float()

    return Xt


def scale_data_within_range(data, ran, x_min, x_max, y_min, y_max):
    """
    Scales your data set within the given range.

    IMPORTANT NOTE: make sure the bounds (x_min, ... y_max) are relatively tight. Else, you're
    effectively shrinking your range. For example, if you generate data between 0 and 4, but you pass
    in 0 and 8 for the bounds, you will really be scaling your data into the first half
    of whatever range you pass in

    :param data: nx2 numpy matrix
    :param ran: the range you want to scale into (a, b)
    :param x_min: min x value
    :param x_max: max x value
    :param y_min: min y value
    :param y_max: max y value
    :return: transformed data
    """

    a, b = ran

    X = data.copy()

    # get the percent of range that the data takes up
    X[:, 0] = (X[:, 0] - x_min) / (x_max - x_min)
    X[:, 1] = (X[:, 1] - y_min) / (y_max - y_min)

    # get corresponding value in range (using the percentages)
    X = (X * (b - a)) + a

    return X


def get_mean_module(name, params):
    if name == 'constant_mean':
        return ConstantMean()
    elif name == 'prior_gp_mean':
        models_and_likelihoods = []
        for param_dict in params:
            state_dict = torch.load(param_dict['state_dict_path'])
            Xt = torch.load(param_dict['Xt_path'])
            scale_factor = param_dict['scale_factor']
            gaussian_lengthscale = param_dict['gaussian_lengthscale']
            min_lengthscale = param_dict['min_lengthscale']
            psi_gamma = param_dict['psi_gamma']
            psi_lambda = param_dict['psi_lambda']
            prior_model = GPClassificationModel(Xt, mean_module=ConstantMean(), min_lengthscale=min_lengthscale)
            prior_model.load_state_dict(state_dict)
            prior_likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)

            models_and_likelihoods.append((prior_model, prior_likelihood))

        prior_mean = CustomMeanModule(models_and_likelihoods, scale_factor=scale_factor,
                                      gaussian_lengthscale=gaussian_lengthscale)

        return prior_mean


def sample_and_train_gp(
        cs,
        grid,
        xx,
        yy,
        mean_module_name='constant_mean',
        mean_module_params=None,
	    sigmoid_type = 'logistic',	
        psi_sigma = .08,	
        psi_gamma=.01,	
        psi_lambda=.01,
        lr=.125,
        num_initial_training_iters=500,
        num_new_points_training_iters=50,
        num_new_points=100,
        beta_for_regularization=1,
        train_on_all_points_after_sampling=False,
        train_on_all_points_iters=1500,
        phi=None,
        print_training_hyperparameters=False,
        print_training_iters=True,
        progress_bar=True,
        min_lengthscale=None,
        random_seed=None,
        calculate_rmse=True,
        calculate_entropy=True,
        calculate_posterior=True,
        initial_Xs=None,
        initial_ys=None,
        sampling_strategy=None,
        strict_labeling=False,
        num_ghost_points=0,
        timepoints=None,
        weight_decay=1e-4,
        kernel_config='new',
        acq_config='new',
        error = 'RMSE',
        orientation = 'vertical'
):
    '''
    initial_Xs is a nx2 numpy array
    initial_ys is a n numpy array
    '''
    # setup timing
    times = []
    if timepoints:
        timepoints_set = set(timepoints)

    # set random seed to get reproducible results
    if random_seed is not None:
        np.random.seed(random_seed)

    # check sampling strategy
    if sampling_strategy == 'active':
        AL_FLAG = True
    elif sampling_strategy == 'random':
        AL_FLAG = False
    else:
        raise Exception('Invalid sampling strategy given. Valid options are "active" or "random".')

    # define the transformation function (usually it does scaling)
    def f(d):
        if phi is not None:
            return phi(d)
        else:
            return d

    # transform the grid
    grid_transformed = transform_dataset(grid, phi=f)

    # We need to loop through and train on each slice of the initial points to calculate rmses or posteriors
    num_initial_points = len(initial_Xs) - num_ghost_points
    num_total_points = num_initial_points + num_new_points

    # make sure you didn't mess up
    if len(initial_Xs) != len(initial_ys):
        raise Exception('Initial Xs and ys do not have the same length.')

    # create copies of the data (so it doesn't affect things outside the function)
    X = np.copy(initial_Xs)
    y = np.copy(initial_ys)

    # store the data during training
    rmse_list = []
    posterior_list = []
    entropy_list = []

    # to calculate data on initial points, need to go through each slice separately
    if calculate_rmse or calculate_posterior:

        # add the posterior GP before training (ie initialized GP)
        # training Xt does not affect initialization, since no training occurs
        Xt_to_initialize = transform_dataset(X[:1, :], phi=f)
        model = GPClassificationModel(Xt_to_initialize,
                                    mean_module=get_mean_module(mean_module_name, mean_module_params),
                                    kernel_config=kernel_config, min_lengthscale=min_lengthscale)
        likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)

        if calculate_posterior:
            posterior_list.append((model, likelihood))

        # i's value will be the index of the non-ghost points
        # for example, if 2 ghost and 10 halton, i=2,3,4,...,11
        for i in range(num_ghost_points, len(initial_Xs)):
            if timepoints:
                startTime = time.perf_counter()

            if print_training_iters:
                print(f'iteration {i - num_ghost_points + 1}/{num_total_points}')

            Xt = transform_dataset(X[:i + 1, :], phi=f)  # notice the X[:i+1, :]
            yt = torch.from_numpy(y[:i + 1]).float()

            model = GPClassificationModel(Xt,
                                          mean_module=get_mean_module(mean_module_name, mean_module_params),
                                          kernel_config=kernel_config, min_lengthscale=min_lengthscale)
            likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
            train_gp_classification_model(model,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=num_initial_training_iters,
                                          lr=lr,
                                          progress_bar=progress_bar,
                                          weight_decay=weight_decay)
            
            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.linear_kernel.variance.data)	
                print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
                print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

            if calculate_rmse:
                Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
                zz = Z.reshape(xx.shape)
                level = (1 - psi_lambda + psi_gamma) / 2
                rmse_list.append(getRMSE(xx, yy, zz, level, cs, error = error, orientation=orientation))

            if calculate_posterior:
                posterior_list.append((model, likelihood))

            curr_num_pts = i - num_ghost_points + 1
            if timepoints and curr_num_pts in timepoints_set:
                times.append((curr_num_pts, time.perf_counter() - startTime))

    startTime = time.perf_counter()

    # next we proceed as usual, training on all points for initial_training_iters
    Xt = transform_dataset(X, phi=f)
    yt = torch.from_numpy(y).float()
    model = GPClassificationModel(Xt,
                                  mean_module=get_mean_module(mean_module_name, mean_module_params),
                                  kernel_config=kernel_config, min_lengthscale=min_lengthscale)
    likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
    train_gp_classification_model(model,
                                  likelihood,
                                  Xt,
                                  yt,
                                  beta=beta_for_regularization,
                                  training_iterations=num_initial_training_iters,
                                  lr=lr,
                                  progress_bar=progress_bar,
                                  weight_decay=weight_decay)
    
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.linear_kernel.variance.data)	
        print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
        print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

    # main loop, repeatedly grabs the next point according to sampling_strategy
    for i in range(num_new_points):

        if print_training_iters:
            print(f'iteration {i + num_initial_points + 1}/{num_total_points}')

        # grab the next point
        if AL_FLAG:
            entropy_grid, best_indices, mean_eval, variance_eval = find_best_entropy(model, likelihood, grid_transformed, xx, Xt, acq_config=acq_config)
        
            if calculate_entropy:
                entropy_list.append(entropy_grid)
            new_x = grid[best_indices[0], :]
            new_y = simulate_labeling(new_x[0], new_x[1], cs, psi_gamma, psi_lambda, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)	
        else:	
            new_x, new_y = random_samples_from_data(grid, cs, psi_gamma, psi_lambda, 1, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)

        # append it to the dataset - MUST BE APPENDED! DON'T PUT ANYWHERE ELSE (or copying over params will fail)
        X_new = np.vstack((X, new_x))
        y_new = np.hstack((y, new_y))

        # transform dataset
        Xt = transform_dataset(X_new, phi=f)
        yt = torch.from_numpy(y_new).float()

        # store the old model state
        old_inducing_points = model.variational_strategy.inducing_points
        old_mean = model.variational_strategy._variational_distribution.variational_mean
        old_covar = model.variational_strategy._variational_distribution.chol_variational_covar
        old_covar_state_dict = model.covar_module.state_dict()
        old_mean_state_dict = model.mean_module.state_dict()
        n = old_inducing_points.shape[0]

        # create new model
        model_new = GPClassificationModel(Xt,
                                          mean_module=get_mean_module(mean_module_name, mean_module_params),
                                          kernel_config=kernel_config, min_lengthscale=min_lengthscale)

        # Copy over hyper parameters
        model_new.covar_module.load_state_dict(old_covar_state_dict)
        model_new.mean_module.load_state_dict(old_mean_state_dict)

        # Copy over variational parameters
        with torch.no_grad():
            # IMPORTANT: we need to make sure the transformation is not local, or else copying over the
            # old points will not be consistent. For example, we can't scale wrt min and max of our data,
            # it has to be wrt a global min/max value (such as the bounds of the grid)
            model_new.variational_strategy.inducing_points[:n, :] = old_inducing_points
            model_new.variational_strategy._variational_distribution.variational_mean[:n] = old_mean
            model_new.variational_strategy._variational_distribution.chol_variational_covar[:n, :n] = old_covar

        # train the model for a few iterations
        try:
            train_gp_classification_model(model_new,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=num_new_points_training_iters,
                                          lr=lr,
                                          progress_bar=progress_bar,
                                          weight_decay=weight_decay)
            model = model_new
            X = X_new
            y = y_new

            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.linear_kernel.variance.data)	
                print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
                print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

        except:
            print("resetting model hypers")
            model_from_scratch = GPClassificationModel(Xt,
                                                       mean_module=get_mean_module(mean_module_name,
                                                                                   mean_module_params),
                                                       kernel_config=kernel_config, min_lengthscale=min_lengthscale)
            train_gp_classification_model(model_from_scratch,
                                          likelihood,
                                          Xt,
                                          yt,
                                          beta=beta_for_regularization,
                                          training_iterations=1500,
                                          lr=lr,
                                          progress_bar=True,
                                          weight_decay=weight_decay)
            model = model_from_scratch
            X = X_new
            y = y_new
            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.linear_kernel.variance.data)	
                print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
                print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)
        if calculate_rmse:
            Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
            zz = Z.reshape(xx.shape)
            level = (1 - psi_lambda + psi_gamma) / 2
            rmse_list.append(getRMSE(xx, yy, zz, level, cs, error = error, orientation=orientation))

        if calculate_posterior:
            posterior_list.append((model, likelihood))

        curr_num_pts = num_initial_points + i + 1
        if timepoints and curr_num_pts in timepoints_set:
            times.append((curr_num_pts, time.perf_counter() - startTime))

    # trains the model from scratch if set to true
    if train_on_all_points_after_sampling:
        Xt = transform_dataset(X, phi=f)
        yt = torch.from_numpy(y).float()
        model = GPClassificationModel(Xt,
                                      mean_module=get_mean_module(mean_module_name, mean_module_params),
                                      kernel_config=kernel_config, min_lengthscale=min_lengthscale)

        train_gp_classification_model(model,
                                      likelihood,
                                      Xt,
                                      yt,
                                      beta=beta_for_regularization,
                                      training_iterations=train_on_all_points_iters,
                                      lr=lr,
                                      progress_bar=progress_bar,
                                      weight_decay=weight_decay)
        print(model.rbf_kernel.lengthscale)

    # this prints out the learned length scale of the final model
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.linear_kernel.variance.data)	
        print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
        print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

    if AL_FLAG:
        return model, likelihood, X, y, rmse_list, entropy_list, posterior_list, times
    else:
        return model, likelihood, X, y, rmse_list, posterior_list, times


def sample_and_train_gp_conjoint(
        cs,
        grid,
        xx,
        yy,
        mean_module_name='constant_mean',
        mean_module_params=None,
	    sigmoid_type = 'logistic',	
        psi_sigma =.08,	
        psi_gamma=.01,	
        psi_lambda=.01,
        lr=.125,
        num_initial_training_iters=500,
        num_new_points_training_iters=50,
        num_new_points=100,
        beta_for_regularization=1,
        train_on_all_points_after_sampling=False,
        train_on_all_points_iters=1500,
        phi=None,
        print_training_hyperparameters=False,
        print_training_iters=True,
        progress_bar=True,
        min_lengthscale=None,
        random_seed=None,
        calculate_rmse=True,
        calculate_entropy=True,
        calculate_posterior=True,
        initial_Xs=None,
        initial_ys=None,
        sampling_strategy=None,
        strict_labeling=False,
        num_ghost_points=0,
        timepoints=None,
        num_tasks=2,
        num_latents=2,
        task_indices=None,
        sampling_method=None,
        weight_decay=1e-4,
        kernel_config='new',
        acq_config='new',
        error = 'RMSE',
        orientation = 'vertical'
):
    '''
    initial_Xs is a nx2 numpy array
    initial_ys is a n numpy array
    '''
    # setup timing
    times = []
    if timepoints:
        timepoints_set = set(timepoints)

    # set random seed to get reproducible results
    if random_seed is not None:
        np.random.seed(random_seed)

    # check sampling strategy
    if sampling_strategy == 'active':
        AL_FLAG = True
    elif sampling_strategy == 'random':
        AL_FLAG = False
    else:
        raise Exception('Invalid sampling strategy given. Valid options are "active" or "random".')

    # define the transformation function (usually it does scaling)
    def f(d):
        if phi is not None:
            return phi(d)
        else:
            return d

    # transform the grid
    grid_transformed = transform_dataset(grid, phi=f)

    # We need to loop through and train on each slice of the initial points to calculate rmses or posteriors
    num_initial_points = len(initial_Xs) - num_ghost_points
    num_total_points = num_initial_points + num_new_points

    # make sure you didn't mess up
    if len(initial_Xs) != len(initial_ys):
        raise Exception('Initial Xs and ys do not have the same length.')

    # create copies of the data (so it doesn't affect things outside the function)
    X = np.copy(initial_Xs)
    y = np.copy(initial_ys)

    # store the data during training
    rmse_list = [[] for _ in range(num_tasks)]
    posterior_list = []
    entropy_list = []

    # to calculate data on initial points, need to go through each slice separately
    if calculate_rmse or calculate_posterior:

        # add the posterior GP before training (ie initialized GP)
        if calculate_posterior:
            model = MTGPClassificationModel(num_latents=num_latents, num_tasks=num_tasks,
                                            kernel_config=kernel_config,min_lengthscale=min_lengthscale)
            likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
            posterior_list.append((model, likelihood))

        # train on initial primer points
        for i in range(num_ghost_points, len(initial_Xs)):
            if timepoints:
                startTime = time.perf_counter()

            if print_training_iters:
                print(f'iteration {i - num_ghost_points + 1}/{num_total_points}')

            # Xt and yt are big_X and big_y, which are the combinations of data from each task
            Xt = transform_dataset(X[:i+1, :], phi=f)  # notice the X[:i+1, :]
            yt = torch.from_numpy(y[:i+1]).float()
            sub_task_indices = task_indices[:i+1]

            model = MTGPClassificationModel(num_latents=num_latents, num_tasks=num_tasks,
                                            kernel_config=kernel_config, min_lengthscale=min_lengthscale)
            likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
            train_gp_classification_model(model, likelihood, Xt, yt, beta=beta_for_regularization, training_iterations=num_initial_training_iters,
                                        progress_bar=progress_bar, task_indices=sub_task_indices,
                                        weight_decay=weight_decay)
            
            if print_training_hyperparameters:	
                print("kernel 0 variance:" , model.linear_kernel.variance.data)	
                print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
                print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

            if calculate_rmse:
                level = (1 - psi_lambda + psi_gamma) / 2
                
                # posterior for all tasks
                zz = evaluate_posterior_mean(model, likelihood, grid_transformed) \
                    .reshape((*xx.shape, num_tasks))

                for j in range(num_tasks):
                    rmse_list[j].append(getRMSE(xx, yy, zz[:,:,j], level, cs[j], error = error, orientation=orientation))

            if calculate_posterior:
                posterior_list.append((copy.deepcopy(model), likelihood))

            curr_num_pts = i - num_ghost_points + 1
            if timepoints and curr_num_pts in timepoints_set:
                times.append((curr_num_pts, time.perf_counter() - startTime))

    startTime = time.perf_counter()

    # next we proceed as usual, training on all points for initial_training_iters
    Xt = transform_dataset(X, phi=f)
    yt = torch.from_numpy(y).float()
    sub_task_indices = task_indices[:Xt.shape[0]]
    model = MTGPClassificationModel(num_latents=num_latents, num_tasks=num_tasks,
                                    kernel_config=kernel_config, min_lengthscale=min_lengthscale)
    likelihood = CustomBernoulliLikelihood(psi_gamma, psi_lambda)
    train_gp_classification_model(model, likelihood, Xt, yt, beta=beta_for_regularization, training_iterations=num_initial_training_iters, 
                                        progress_bar=progress_bar, task_indices = sub_task_indices,
                                        weight_decay=weight_decay)
        
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.linear_kernel.variance.data)	
        print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
        print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

    # main loop, repeatedly grabs the next point according to sampling_strategy
    for i in range(num_new_points):

        if print_training_iters:
            print(f'iteration {i + num_initial_points + 1}/{num_total_points}')

        # grab the next point 
        if AL_FLAG:

            if sampling_method == 'alternating':
                
                task_index = i % num_tasks
                _, best_indices, _, _ = find_best_entropy(model, likelihood, grid_transformed, xx, Xt, task_index = task_index, acq_config=acq_config)
                best_index = best_indices[0]
            
            if sampling_method == 'unconstrained':
                best_entropy = 0
                best_index = -1
                task_index = -1

                for j in range(num_tasks): # this for loop finds which task has the max entropy
                    entropy_grid, best_indices = find_best_entropy(model, likelihood, grid_transformed, xx, Xt, j, acq_config=acq_config)
                    curr_best_index = best_indices[0]
                    task_entropy = entropy_grid.flatten()[curr_best_index].item()

                    print("entropy for task", j, "is", task_entropy)

                    # save data for best entropy point
                    if task_entropy > best_entropy:
                        best_entropy = task_entropy
                        best_index = curr_best_index
                        task_index = j
            
            new_x = grid[best_index, :]
            new_y = simulate_labeling(new_x[0], new_x[1], cs[task_index], psi_gamma, psi_lambda, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)	
        else:	
            # TODO: fix later
            new_x, new_y = random_samples_from_data(grid, cs, psi_gamma, psi_lambda, 1, strict=strict_labeling, sigmoid_type=sigmoid_type, psi_sigma = psi_sigma)

        # append new data
        X_new = np.vstack((X, new_x))
        y_new = np.hstack((y, new_y))
        task_indices_new = torch.cat((task_indices, torch.tensor([task_index]))) # append the task_indices as well

        # transform dataset
        Xt = transform_dataset(X_new, phi=f)
        yt = torch.from_numpy(y_new).float()
        task_indices = task_indices_new

        # train the model for a few iterations
        try:
            train_gp_classification_model(model, likelihood, Xt, yt, beta=beta_for_regularization, training_iterations=num_new_points_training_iters, 
                                        progress_bar=progress_bar, task_indices=task_indices,
                                        weight_decay=weight_decay)
            X = X_new
            y = y_new
        except:
            print("resetting model hypers")
            model_from_scratch = MTGPClassificationModel(num_latents=num_latents, num_tasks=num_tasks,
                                                         kernel_config=kernel_config, min_lengthscale=min_lengthscale)
            train_gp_classification_model(model_from_scratch, likelihood, Xt, yt, beta=beta_for_regularization, training_iterations=num_new_points_training_iters, 
                                        progress_bar=progress_bar, task_indices = task_indices,
                                        weight_decay=weight_decay)

            model = model_from_scratch
            X = X_new
            y = y_new

        if print_training_hyperparameters:	
            print("kernel 0 variance:" , model.linear_kernel.variance.data)	
            print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
            print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

        if calculate_rmse:
            level = (1 - psi_lambda + psi_gamma) / 2

             # posterior for all tasks
            zz = evaluate_posterior_mean(model, likelihood, grid_transformed) \
                .reshape((*xx.shape, num_tasks))

            for j in range(num_tasks):
                rmse_list[j].append(getRMSE(xx, yy, zz[:,:,j], level, cs[j], error = error, orientation=orientation))

        a = time.perf_counter()
        if calculate_posterior:
            posterior_list.append((copy.deepcopy(model), likelihood))

        grid_transformed = transform_dataset(grid, phi=f)

        curr_num_pts = num_initial_points + i + 1
        if timepoints and curr_num_pts in timepoints_set:
            times.append((curr_num_pts, time.perf_counter() - startTime))

    # trains the model from scratch if set to true
    if train_on_all_points_after_sampling:
        Xt = transform_dataset(X, phi=f)
        yt = torch.from_numpy(y).float()
        model = MTGPClassificationModel(Xt,
                                      mean_module=get_mean_module(mean_module_name, mean_module_params),
                                      kernel_config=kernel_config, min_lengthscale=min_lengthscale
                                      )
        train_gp_classification_model(model,
                                      likelihood,
                                      Xt,
                                      yt,
                                      beta=beta_for_regularization,
                                      training_iterations=train_on_all_points_iters,
                                      lr=lr,
                                      progress_bar=progress_bar,
                                      weight_decay=weight_decay)

    # this prints out the learned length scale of the final model
    if print_training_hyperparameters:	
        print("kernel 0 variance:" , model.linear_kernel.variance.data)	
        print("kernel 1 outputscale:" , model.covar_module.outputscale.data)	
        print("kernel 1 lengthscale:" , model.rbf_kernel.lengthscale.data)

    if AL_FLAG:
        return model, likelihood, X, y, task_indices, rmse_list, entropy_list, posterior_list, times
    else:
        return model, likelihood, X, y, task_indices, rmse_list, posterior_list, times


def interactive_gif_gp_posterior(
        my_dict,
        ax,
        model,
        likelihood,
        X,
        y,
        title='Visualization',
        latent_color='purple',
        mean_color='turquoise',
        level=0.5,
        xticks_labels=np.array([1, 4, 16, 64]),
        yticks_labels=np.array([1, 10, 100, 1000])
):
    xx = my_dict['xx']
    yy = my_dict['yy']
    left = my_dict['left']
    right = my_dict['right']
    cs = my_dict['cs']
    x_min = my_dict['x_min']
    x_max = my_dict['x_max']
    y_min = my_dict['y_min']
    y_max = my_dict['y_max']
    xs = my_dict['xs']
    ys = my_dict['ys']
    grid = my_dict['grid']
    f = my_dict['f']

    grid_transformed = transform_dataset(grid, phi=f)

    # get the predictions on the eval grid
    Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
    zz = Z.reshape(xx.shape)

    # plot the contour field
    resolution = 111
    ax.pcolormesh(xx, yy, zz, cmap='gist_gray', vmin=0, vmax=1)

    # plot the training data
    ax.scatter(X[y == 1, 0].reshape(-1),
               X[y == 1, 1].reshape(-1), label='Success', marker='.', c='blue')
    ax.scatter(X[y == 0, 0].reshape(-1),
               X[y == 0, 1].reshape(-1), label='Failure', marker='.', c='red')

    # plot the spline
    latent_x1 = np.linspace(left, right, 750)
    latent_x2 = cs(latent_x1)
    ax.plot(latent_x1, latent_x2, color=latent_color)

    # plot the level curve
    ax.contour(xx, yy, zz, levels=[level], colors=mean_color)

    # specify the tick marks here
    # _labels are the numbers you want to display
    # _values are the underlying values corresponding to these labels
    # in this case, the underlying values are log10 of the labels
    # _labels and _values must be the same length
    xticks_values = logFreq().forward(xticks_labels)

    yticks_values = logContrast().forward(1 / yticks_labels)

    ax.set_xticks(xticks_values, xticks_labels)
    ax.set_yticks(yticks_values, yticks_labels)

    # fit to the grid
    x_padding = (x_max - x_min) / (2 * (xs - 1))
    y_padding = (y_max - y_min) / (2 * (ys - 1))
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # title and axis labels
    ax.set_title(title, fontdict={'fontsize': 12})
    ax.set_xlabel('Spatial Frequency (cyc/deg)', fontdict={'fontsize': 12})
    ax.set_ylabel('Contrast', fontdict={'fontsize': 12})

def create_and_save_plots(
        results_dict,
        path,
        title,
        start_index=0,
        latent_color='#cf30cf',
        mean_color='#40E1D0',
        xticks_labels=[1, 4, 16, 64],
        yticks_labels=[1, 0.1, 0.01, 0.001]
):
    # load data from results 
    xx = results_dict['xx']
    yy = results_dict['yy']
    X = results_dict['X']
    y = results_dict['y']
    cs = results_dict['cs']
    psi_gamma = results_dict['psi_gamma']
    psi_lambda = results_dict['psi_lambda']
    x_min = results_dict['x_min']
    x_max = results_dict['x_max']
    y_min = results_dict['y_min']
    y_max = results_dict['y_max']
    xs = results_dict['xs']
    ys = results_dict['ys']
    grid = results_dict['grid']
    f = results_dict['f']
    posterior_list = results_dict['posterior_list']

    level = (1 - psi_lambda + psi_gamma) / 2

    # font type
    plt.rcParams['font.family'] = 'sans-serif'

    # data point configs
    marker_size= 60                 # marker size specified in pts

    diamond_line_width = .8         # line width specified in pts
    diamond_size = .4               # scaling constant
    diamond_width = .152            # specified in cycles per degree
    diamond_height = .10            # specified in log10(contrast)

    success_marker = '+'  
    plus_line_width= .8

    diamond_color = 'red'
    diamond_fill_color = 'none'     # 'none' for transparent diamonds
    plus_color = 'blue'
              
    # dashed line style for predicted mean
    mean_linestyle= 'dashed'
    mean_dash_list = [(0, (10.0, 3.0))]

    # ticks
    xticks_values = logFreq().forward(np.array(xticks_labels))
    yticks_values = logContrast().forward(np.array(yticks_labels))
    axis_tick_params = {
        'axis':'both', 
        'which':'major', 
        'direction':'out'
    }

    # colorbar
    colorbar_ticks = [0, .25, .5, .75, 1]
    colorbar_labels = [str(tick) for tick in colorbar_ticks]
    colorbar_label_pad = -50

    plt.figure(figsize=(6,4))

    for i, (model, likelihood) in enumerate(posterior_list):

        # use same figure but cleared
        plt.clf()

        # start with 0 datapoints
        curr_X = X[start_index:start_index+i, :]
        curr_y = y[start_index:start_index+i]

        # plot predictive posterior
        grid_transformed = transform_dataset(grid, phi=f)
        Z = evaluate_posterior_mean(model, likelihood, grid_transformed)
        zz = Z.reshape(xx.shape)
        posterior = plt.pcolormesh(xx, yy, zz, cmap='gist_gray', vmin=0, vmax=1)

        # add colorbar
        cbar = plt.colorbar(posterior)
        cbar.set_ticks(colorbar_ticks)
        cbar.set_ticklabels(colorbar_labels)
        cbar.set_label("Detection Probability", fontsize=10, labelpad=colorbar_label_pad)

        # plot ground truth
        latent_x1 = np.linspace(x_min, x_max, 750)
        latent_x2 = cs(latent_x1)
        plt.plot(latent_x1[latent_x2 > y_min], latent_x2[latent_x2 > y_min], color=latent_color)

        # plot predicted threshold
        CS = plt.contour(xx, yy, zz, levels=[level], colors=mean_color, linestyles=mean_linestyle)
        
        for c in CS.collections:
            c.set_dashes(mean_dash_list)

        # plot datapoints
        # diamonds (failure)
        x1_failure = curr_X[curr_y == 0, 0].reshape(-1)
        x2_failure = curr_X[curr_y == 0, 1].reshape(-1)

        # coordinates for diamond corners
        diamond_corners = np.array([
                                  [x1_failure, x2_failure + diamond_height*diamond_size],
                                  [x1_failure + diamond_width*diamond_size, x2_failure],
                                  [x1_failure, x2_failure - diamond_height*diamond_size],
                                  [x1_failure - diamond_width*diamond_size, x2_failure]])
        
        plt.fill(diamond_corners[:, 0], diamond_corners[:, 1], color=diamond_fill_color,
                    edgecolor=diamond_color, linewidth=diamond_line_width)

        # plusses (success)
        x1_success = curr_X[curr_y == 1, 0].reshape(-1)
        x2_success = curr_X[curr_y == 1, 1].reshape(-1)
        plt.scatter(x1_success, x2_success, marker=success_marker, s=marker_size, color=plus_color, 
                    linewidth=plus_line_width)

        # tick marks
        plt.xticks(xticks_values, xticks_labels)
        plt.yticks(yticks_values, yticks_labels)
        plt.tick_params(**axis_tick_params, labelsize=10)

        # add padding so points at limits are not cut off
        x_padding = (x_max - x_min) / (2 * (xs - 1))
        y_padding = (y_max - y_min) / (2 * (ys - 1))
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

        # title and axis labels
        plt.title(f"{title} ({i})", fontsize=12)
        plt.xlabel('Spatial Frequency (cyc/deg)', fontsize=12)
        plt.ylabel('Contrast', fontsize=12)
        
        Path(path).mkdir(parents=True, exist_ok=True)
        plt.savefig(path + f'{i}-{title}', bbox_inches='tight')
    
    plt.close()

def create_gp_plot(
        ax,
        results_dict,
        title="",
        marker_size=60,
        marker_line_width=0.8,
        line_width=2,
        title_font_size=12,
        label_font_size=10,
        tick_font_size=10,
        xticks_labels=[1, 4, 16, 64],
        yticks_labels=[1, 0.1, 0.01, 0.001],
        vmin=0,
        vmax=1,
        labelx = True,
        labely = True
):
    # load data from results 
    xx = results_dict['xx']
    yy = results_dict['yy']
    zz = results_dict['zz']
    X = results_dict['X']
    y = results_dict['y']
    cs = results_dict['cs']
    psi_gamma = results_dict['psi_gamma']
    psi_lambda = results_dict['psi_lambda']
    x_min = results_dict['x_min']
    x_max = results_dict['x_max']
    y_min = results_dict['y_min']
    y_max = results_dict['y_max']
    xs = results_dict['xs']
    ys = results_dict['ys']
    
    # add important plot details to axes
    add_posterior(ax, xx, yy, zz, vmin=vmin, vmax=vmax)
    add_ground_truth(ax, cs, x_min, x_max, y_min, line_width)
    add_threshold(ax, xx, yy, zz, psi_gamma, psi_lambda, line_width)
    add_datapoints(ax, X, y, marker_size, marker_line_width=marker_line_width)

    configure_subplot(ax, x_min, x_max, y_min, y_max, \
            xticks_labels, yticks_labels, title, \
            title_font_size, label_font_size, tick_font_size, labelx=labelx, labely = labely)
    
    return ax

def create_gp_progress_plot(
        ax,
        results_dict,
        model,
        likelihood,
        grid_transformed,
        title="",
        line_width=2,
        title_font_size=12,
        label_font_size=10,
        tick_font_size=10,
        xticks_labels=[1, 4, 16, 64],
        yticks_labels=[1, 0.1, 0.01, 0.001],
        vmin=0,
        vmax=1,
        color='#40E1D0',
        line_style='dashed',
        alpha = 1,
        labelx = True,
        labely = True,
        num_tasks = None,
        conjoint_index = None
):
    # load data from results 
    xx = results_dict['xx']
    yy = results_dict['yy']
    cs = results_dict['cs']
    psi_gamma = results_dict['psi_gamma']
    psi_lambda = results_dict['psi_lambda']
    x_min = results_dict['x_min']
    x_max = results_dict['x_max']
    y_min = results_dict['y_min']
    y_max = results_dict['y_max']

    if num_tasks is not None:
        # posterior for all tasks
        zz = evaluate_posterior_mean(model, likelihood, grid_transformed).reshape((*xx.shape, num_tasks))
        # add important plot details to axes
        add_threshold(ax, xx, yy, zz[:,:,conjoint_index], psi_gamma, psi_lambda, line_width, color=color, line_style=line_style, alpha=alpha)
    else:
        zz = evaluate_posterior_mean(model, likelihood, grid_transformed).reshape(xx.shape)
        add_threshold(ax, xx, yy, zz[:,:], psi_gamma, psi_lambda, line_width, color=color, line_style=line_style, alpha=alpha)

    configure_subplot(ax, x_min, x_max, y_min, y_max, \
            xticks_labels, yticks_labels, title, \
            title_font_size, label_font_size, tick_font_size, labelx=labelx, labely = labely)
    
    return ax

def configure_subplot(
    ax,
    x_min,
    x_max,
    y_min,
    y_max,
    xticks_labels=[1, 4, 16, 64],
    yticks_labels=[1, 0.1, 0.01, 0.001],
    title="",
    title_font_size=12,
    label_font_size=10,
    tick_font_size=10,
    x_padding = 0.03,
    y_padding = 0.015,
    labelx = True,
    labely = True
):

    # set tick marks
    xticks_values = logFreq().forward(np.array(xticks_labels))
    yticks_values = logContrast().forward(np.array(yticks_labels))
    axis_tick_params = {
        'axis': 'both', 
        'which': 'major', 
        'direction': 'out'
    }

    ax.set_xticks(xticks_values)
    ax.set_xticklabels(xticks_labels)
    ax.set_yticks(yticks_values)
    ax.set_yticklabels(yticks_labels)
    ax.tick_params(**axis_tick_params, labelsize=tick_font_size)

    # add padding so points at limits are not cut off
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # title and axis labels
    ax.set_title(title, fontsize=title_font_size)
    if labelx:
        ax.set_xlabel('Spatial Frequency (cyc/deg)', fontsize=label_font_size)
    if labely:
        ax.set_ylabel('Contrast', fontsize=label_font_size)

def add_datapoints(ax, X, y, marker_size=60, marker_line_width=0.8, success_color='blue', failure_color='red'):
    # make each diamonds a similar size as a plus
    failure_marker_size = 0.7 * marker_size  
    success_marker_size = marker_size

    X_success = X[y == 1, :]
    x1_success = X_success[:, 0].reshape(-1)
    x2_success = X_success[:, 1].reshape(-1)
    ax.scatter(x1_success, x2_success, marker='+', s=success_marker_size, color=success_color,
            linewidth=marker_line_width)

    X_failure = X[y == 0, :]
    x1_failure = X_failure[:, 0].reshape(-1)
    x2_failure = X_failure[:, 1].reshape(-1)
    ax.scatter(x1_failure, x2_failure, s=failure_marker_size, marker='d', facecolor='none',
            edgecolor=failure_color, linewidth=marker_line_width)
    
    return ax, X_success, X_failure


def add_ground_truth(ax, cs, x_min, x_max, y_min, line_width=2, color='#cf30cf'):
    latent_x1 = np.linspace(x_min, x_max, 750)
    latent_x2 = cs(latent_x1)
    ground_truth_curve = ax.plot(latent_x1[latent_x2 > y_min], latent_x2[latent_x2 > y_min], linewidth=line_width, color=color)

    return ax, ground_truth_curve

def add_posterior(ax, xx, yy, zz, vmin=0, vmax=1):
    # plot predictive posterior
    posterior = ax.pcolormesh(xx, yy, zz, cmap='gist_gray', vmin=vmin, vmax=vmax)
    return ax, posterior

def add_colorbar(ax, posterior, label_font_size=12, tick_font_size=10, pad=0.07, label_pad=-45):
    cbar = plt.colorbar(posterior, ax=ax, pad=pad, format='%g')
    
    cbar.set_label("Detection Probability", fontsize=label_font_size, labelpad=label_pad)  
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.ax.tick_params(labelsize=tick_font_size)

    return ax, cbar


def add_threshold(ax, xx, yy, zz, psi_gamma, psi_lambda, line_width=2, color='#40E1D0', line_style='dashed', alpha = 1, dash_list=[(0, (10.0, 3.0))]):
    level = (1 - psi_lambda + psi_gamma) / 2
    threshold_curve = ax.contour(xx, yy, zz, levels=[level], colors=color, linestyles=[line_style], linewidths=line_width, alpha = alpha)
    
    if line_style == 'dashed':
        for curve in threshold_curve.collections:
            curve.set_dashes(dash_list)

    return ax, threshold_curve



def create_gif(path):
    image_filenames = [file for file in os.listdir(path) if file.endswith('.png')]
    image_filenames = sorted(image_filenames, key=lambda fn: int(fn.split('-')[0]))

    images = []
    for fn in image_filenames:
        images.append(Image.open(path + fn))

    images[0].save(
        path + 'summary_gif.gif',
        save_all=True,
        append_images=images[1:],
        duration=500,
        loop=0,
    )


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_directory_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json_from_file(path):
    with open(path, 'r') as file:
        jsonobj = json.load(file)
    return jsonobj