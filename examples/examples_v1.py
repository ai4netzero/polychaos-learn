
""" Transformation of features for Polynomial Chaos Expansion
"""

# Author: Ahmed ElSheikh <a.elsheikh.@hw.ac.uk>
#         Alexander Tarakanov <a.tarakanov@hw.ac.uk>


import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNetCV

from pclearn import OrthogonalPolynomialsFeatures


def ish_2d(x):
    # Ishigami Function
    x_0 = x[:, 0]
    x_1 = x[:, 1]
    term1 = 2.0 * x_0 + 0.1 * np.sin(2.0 * x_0)
    term2 = -3.28 * x_1 + 7.0 * np.multiply(np.sin(x_1), np.sin(x_1)) - 3.5
    y = term1 + term2
    return y


def load_ish_data(n_samples, manual_seed=12345):
    # generate data for modified Ishigamy function from the paper from arXiv 1808.03216
    x_all = np.random.rand(n_samples, 2)
    x_all = 2.0 * x_all - 1.0
    y_all = ish_2d(np.pi * x_all)
    return train_test_split(x_all, y_all, test_size=0.33, random_state=manual_seed)


if __name__ == '__main__':
    do_plot = True

    manual_seed = 123456  # random.randint(1, 10000)  # fix seed randomly
    random.seed(manual_seed)
    np.random.seed(manual_seed)

    # what are the other ideas that you have?
    # I can test the code on CO2 injection data
    # on on the Ishigami function data

    # load data set
    n_samples = 2000  # number of training points (number of test points is the same)
    X_train, X_test, y_train, y_test = load_ish_data(n_samples, manual_seed)
    n_samples_train, n_features = X_train.shape
    # define solver parameters
    tol_eps = 1.0e-6  # tolerance
    n_cv_folds = 5
    alphas = np.logspace(-5, 1, 5)
    l1_ratio = np.linspace(0.05, 0.95, 10)
    n_max_iter = 500  # number of iteratons

    fitting_method = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
                                  max_iter=n_max_iter, tol=tol_eps, cv=n_cv_folds,
                                  fit_intercept=False, selection='random')

    # define interpolation parameters
    poly_degree = 20  # degree of polynomial
    poly_type = ['legendre'] * n_features  # could be different e.g. ['legendre', 'hermite']

    reduction = False
    dlist = []
    truncated_dim = 20
    single_iter_dim = 10
    niter = 500
    nrand = 50

    orthopoly = OrthogonalPolynomialsFeatures(
        degree=poly_degree, dlist=dlist, poly_type=poly_type,
        reduction=reduction, basis_dim=truncated_dim,
        single_iter_dim=single_iter_dim, niter=niter, nrand=nrand,
        clf=fitting_method)

    # Parameters of the regression solver

    print('Ready to solve')

    regression_pipeline = Pipeline(steps=[
        ('orthopoly', orthopoly),
        ('fitting_method', fitting_method)])

    param_grid = {
        'orthopoly__degree': [3, 5, 7],
        'fitting_method__alpha': alphas,
    }

    print('Solving linear problem')
    regression_pipeline.fit(X_train, y_train)
    print('Linear problem has been solved')
    y_train_pred = regression_pipeline.predict(X_train)
    y_test_pred = regression_pipeline.predict(X_test)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)

    print('MSE train: {} vs. test {}'.format(mse_train, mse_test))

    if do_plot:
        fig = plt.figure(figsize=(16, 10))
        ax1 = fig.add_subplot(121)
        plt.scatter(y_train, y_train_pred)
        ax1.set_aspect('equal', adjustable='box')
        plt.grid()

        ax2 = fig.add_subplot(122, sharey=ax1)
        plt.scatter(y_test, y_test_pred)
        ax2.set_aspect('equal', adjustable='box')
        plt.grid()
        plt.title('Train versus test results', fontsize=24)
        plt.show()
