""" Transformation of features for Polynomial Chaos Expansion (PCE)
"""

# Author: Ahmed ElSheikh <a.elsheikh@hw.ac.uk>
#         Alexander Tarakanov <a.tarakanov@hw.ac.uk>


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.linear_model import ElasticNetCV
import scipy as sp


def init_index_matrix_cumdeg(max_degree, n_features):
    """Function for recurrent calculation of matrix for feature transformation.
    The matrix feature transformation or index matrix is table of degrees
    of monomials of degree d of (n+1) variables.

    Input parameters are assumed to be integers.
    Elements of output matrix are integers.

    Output is the matrix with integer elements

    Parameters
    ----------
    max_degree: integer
    maximal degree of polynomial

    n_features: integer
    number of features (variables)
    """

    if n_features == 0:
        raise Exception('Number of feature have to be at least 1')

    if max_degree == 1:  # simple case - Constant polynomials
        return np.eye(n_features, dtype=np.int64)

    if n_features == 1:  # simple case - one variable
        return np.array([[max_degree]], dtype=np.int64)

    n_combinations = int(sp.special.binom(n_features + max_degree - 1, max_degree))
    ind_mat = np.zeros([n_combinations, n_features], dtype=np.int64)

    start_row_ = 0
    for deg_idx_ in np.arange(max_degree + 1):
        # recurrent evaluation
        A = init_index_matrix_cumdeg(deg_idx_, n_features - 1)
        n_rows, n_columns = A.shape
        for row_idx_ in np.arange(n_rows):
            ind_mat[start_row_ + row_idx_, 0] = max_degree - deg_idx_
            for col_idx_ in np.arange(1, n_columns + 1):
                ind_mat[start_row_ + row_idx_, col_idx_] = A[row_idx_, col_idx_ - 1]
                if(start_row_ + row_idx_ >= ind_mat.shape[0]):
                    raise Exception('Segmentation fault in rows of Init_Index_Matrix')
                if(col_idx_ >= ind_mat.shape[1]):
                    raise Exception('Segmentation fault in columns of Init_Index_Matrix')
        start_row_ = start_row_ + n_rows
    return ind_mat


def init_index_matrix(max_degree, n_features, dlist=[]):
    """ Update of the matrix of the feature transformation
    There are four things that are made with the index matrix in this step

    1) Reordering of the matrix with respect to the total degree of
        basis polynomial
    2) Removal of all rows that violate the constraint on maximal degree of
    single-variable polynomial
    3) Removal of the first column with the value of total degree
    4) Transposition of the matrix

    The output is the matrix with integer elements

    Parameters
    ----------
    max_degree: integer
    maximal degree of product of basis single-variable polynomials
    n_features: integer
    number of variables or number of features
    dlist: list of integers
    list of maximal degrees for basis single-variable polynomials in products
    """

    ind_mat = init_index_matrix_cumdeg(max_degree, n_features + 1)

    # Reordering of the rows os the matrix
    ind_sorted = np.argsort(-ind_mat[:, 0])  # reordering of indices
    ind_mat1 = ind_mat[ind_sorted, :]
    row_list = []  # list of rows to leave

    # Constraint on the maximal degree of single-variable basis polynomial
    if(len(dlist) > 0):
        n_rows_ = ind_mat1.shape[0]
        for k0 in np.arange(n_rows_):
            var0 = 1
            for k1 in np.arange(len(dlist)):
                if ind_mat1[k0, 1 + k1] > dlist[k1]:
                    var0 = 0
            if var0 == 1:
                row_list.append(k0)
        ind_mat2 = ind_mat1[row_list, :]
        # removing column with total degrees and transposition
        return np.transpose(ind_mat2[:, 1:n_features + 1])
    # removing column with total degrees and transposition
    return np.transpose(ind_mat1[:, 1:n_features + 1])


def reduce_dimension(XP, y, combinations_mat, poly_type_list,
                     basis_dim, single_iter_dim,
                     n_iter, n_rand, clf):
    ''' Reduction of the size of the combinations_mat.
    The reduction is made by keeping some of the columns of the matrix
    and by deleting others.
    The reduction is made based on the input data.
    Number of columns to leave is basis_dim

    Parameters
    ----------

    XP: ndarray of real numbers
    matrix of transformed input parameters

    y: one-dimensional ndarray
    array of values of output data

    combinations_mat: 2d array of integers
    matrix for feature transformation

    poly_type: list of strings
    list of the types of orthogonal poynomials for each of the variables
    each element of the list can take one of those values:
    'Legendre', 'Hermite Probabilistic', 'Hermite Physicist',
    'Chebyshev', 'Laguerre'
    empty list is equivalent to list with all elements equal to
    'Hermite Probabilistic'


    basis_dim: integer
    number of columns of combinations_mat to save

    single_iter_dim: integer
    number of basis functions to add in a single iteration

    niter: integer
    number of iterations

    nrand: integer
    number of random instances for calculation of variances
    '''
    # transformation of the data for calculation of
    assert clf is not None
    float_eps = 1.0e-6
    n_input_features, n_output_features = combinations_mat.shape

    # Orthogonal single_variable polynomials are not orthonormal
    # Here normalization is made in such a way that
    # all polynomials have unit variance

    # types of polynomials are not specified
    # Hermite probabilistic polynomials are used as a default option
    if len(poly_type_list) == 0:
        poly_type_list = ['hermite_probabilistic'] * n_input_features
        # use norm for Hermite Polynomials
        print('Polynomial types are not specified, use default hermite_probabilistic poynomials')

    if len(poly_type_list) < n_input_features:
        raise Exception('Number of polynomial types is not equal to then number of features')
    if len(poly_type_list) > n_input_features:
        print('Number of polynomial types is greater that the number of features, use the first n elements')
        poly_type_list = poly_type_list[:n_input_features]

    normf = np.ones(n_output_features, dtype=np.float64)
    for idx_, poly_type in enumerate(poly_type_list):
        poly_type = poly_type_list[idx_]
        if poly_type == 'legendre':
            normf *= np.sqrt(1.0 / (2.0 * combinations_mat[idx_, :] + 1.0))
        elif poly_type == 'hermite_probabilistic':
            normf *= np.sqrt(1.0 * sp.special.factorial(combinations_mat[idx_, :]))
        elif poly_type == 'hermite_physicist':
            normf *= np.sqrt((2.0 ** combinations_mat[idx_, :]) * sp.special.factorial(combinations_mat[idx_, :]))
        elif poly_type == 'chebyshev':
            mask_ = (combinations_mat[idx_, :] > 0)
            normf[mask_] *= np.sqrt(0.5)
        elif poly_type_list[idx_] == 'Laguerre':
            # Laguerre polynomials have unit norm.
            # Therefore, nothing should be done.
            pass
        else:
            raise Exception('Polynomial type is not recognized')

    n_samples = XP.shape[0]

    y_perturbed = np.tile(y.reshape(-1, 1), (1, n_rand)) + np.random.normal(0.0, 1.0, (n_samples, n_rand))
    corr_coefs_U = np.dot(y_perturbed.T, XP)
    corr_coefs_V = np.dot(y.T, XP)
    corr_coefs = corr_coefs_U - corr_coefs_V

    y_noise_sens = np.sqrt(np.mean(np.multiply(corr_coefs, corr_coefs), axis=0))
    y_noise_sens = np.divide(y_noise_sens, normf)

    XP_sens = np.multiply(XP,XP)
    XP_sens = np.divide(XP_sens, np.tile(np.multiply(normf, normf), (XP_sens.shape[0], 1))) - 1.0
    X_noise_sens = np.sqrt(np.mean(np.multiply(XP_sens, XP_sens), axis=0))
    X_noise_sens = np.reshape(X_noise_sens, [X_noise_sens.size]) + float_eps

    # Iterative ranking procedure
    y_residual = y.copy()
    # list of rows of combinations_mat that will be saved
    ind_list = []
    for itr_ in np.arange(n_iter):
        # compute correlation
        if len(ind_list) >= basis_dim:
            break
        # Calculation of covariance coefficients with residual
        covar_coef = np.dot(y_residual, XP) / n_samples
        # apply penalty for the sensitivity to noise in y data
        X_term = np.multiply(X_noise_sens, X_noise_sens)
        y_term = np.multiply(y_noise_sens, y_noise_sens)
        rank_coef = np.divide(covar_coef, np.sqrt(float_eps + X_term + y_term)) + float_eps
        # transform for better sepration of values in ranking coefficient
        rank_coef = 2.0 * rank_coef / (np.amax(np.abs(covar_coef)) + np.amin(np.abs(covar_coef)) + float_eps)
        rank_coef = np.log(1.0 / (1.0 + 1.0 / np.multiply(rank_coef, rank_coef)))
        # reordering with respect to the magnitude of rankijng coefficient
        # Highly ranked come first
        ind_list1 = np.argsort(rank_coef)[-single_iter_dim:]

        # add elements to the the list of rows that have to be saved
        # ind_list2 = list(set(ind_list).union(set(ind_list1)))
        # ind_list2 = np.union1d(ind_list, ind_list1).astype(np.int64)
        # print(ind_list2.dtype)

        ind_list2 = list(set(ind_list).union(set(ind_list1)))
        if len(ind_list2) <= basis_dim:
            ind_list = list(set(ind_list).union(set(ind_list1)))
        else:
            for k in range(single_iter_dim):
                if (len(ind_list) < basis_dim):
                    ind_list = list(set(ind_list).union(set([ind_list1[k]])))

        # Solving for small regression problem for upddate the residual
        XP_iter = XP[:, ind_list1]
        clf.fit(XP_iter, y_residual)
        # y_residual -= clf_esnet.predict(XP_iter)
        y_comp = clf.predict(XP_iter)
        y_residual -= y_comp
    '''
    if(len(ind_list) < basis_dim):
            # if numer of selected components is less then desired -
            # take neccesary number from original combinations_mat
            all_ind_list = list(range(n_output_features))
            remaining_list = list(set(all_ind_list).difference(set(ind_list)))
            n_to_add = basis_dim - len(ind_list)
            ind_list = set(ind_list).union(set(remaining_list[0:n_to_add]))
            ind_list = list(ind_list)
    '''
    return combinations_mat[:, ind_list]


class OrthogonalPolynomialsFeatures(BaseEstimator, TransformerMixin):
    """Polynomial Chaos Expansion (PCE)

    This class is designed for preparation of features for PCE.
    The default option is transformation of features for classic PCE
    for given number of variables (it is derived from dimensions of the data)
    and for given polynomial degree.

    In addition to classical PCE, this library allows one to use different
    types of orthogonal polynomials for different variables.
    Types of polynomials should be specified in poly_type.

    It also possible to consider restrictions not only for
    the overall polynomial degree
    but for degrees with respect to each of the features (variables).
    Maximum degrees for each of the features should be specified in dlist.

    There is also an option for reduced PCE.
    The reduced form of PCE works only with the part of the features of
    classical PCE. The desired number of output features is specified in
    basis_dim.
    The reduction is an iterative process that is based on the input data y and X.
    The reduction is performed if the value of the flag 'reduction' is True.
    Overall number of iteration in ranking procedure is niter.
    Maximum number of features that can be added at each iteration is
    single_iter_dim
    The number of random instances that are used in intermediate calculations
    is specified in nrand


    Parameters
    ----------
    degree: integer
        maximum degree pf basis polynomials used for PCE
    dlist: list of non-negative integers
        list of non-negative integers that set upper limit for
        the degree of single_variables factors
        in basis polynomials.
        The default value is empty list. In this case no constraints are applied
        If number of constraints is less then number of
        input features (variables),
        then constraints will be applied for first len(dlist) variables only.
    poly_type: list of strings
        each element of the string can take one of the following values:
        'Legendre', 'Hermite Probabilistic', 'Hermite Physicist',
        'Chebyshev' or 'Laguerre'
        Default value is empty list.
        In this case Hermite Probabilistic polynomials are be used
        for all features.
        If length of poly_type is less then number of features,
        then for first len(poly_type) elements polynomials
        of the assigned types are be used.
        For the remaining features 'Hermite Probabilistic' polynomials are used.
    reduction: True or False
        Flag for reduced PCE.
        If True - reduced PCE is used.
        Otherwise, features for full PCE are generated
        Default value is False.
    basis_dim: integer
        dimension of the basis for full PCE.
        Default value is 20.
    single_iter_dim: integer
        number of basis functions selected for dimension reduction
        at given iteration.
        Default value 5.
    niter: integer
        number of iterations in dimension reduction procedure.
        Default value is 20.

    nrand: integer
        number of random instances. It used for intermediate calculations
        inside the dimension reduction procedure. It is needed for numerical
        estimation of variance based on MC.
        Default value is 100.
    """

    def __init__(self, degree=1, dlist=[], poly_type=[], reduction=False,
                 basis_dim=20, single_iter_dim=5, niter=20, nrand=100,
                 clf=None):
        self.degree = degree
        self.dlist = dlist
        self.poly_type = poly_type
        self.reduction = reduction
        self.basis_dim = basis_dim
        self.single_iter_dim = single_iter_dim
        self.niter = niter
        self.nrand = nrand
        self.clf = clf

        self.n_input_features = 0
        self.n_output_features = 0
        self.combinations_mat = None

        # print('Polynomial degree in init section')
        # print(self.degree)

    def fit(self, X, y=None):
        """Initialization of combinations matrix for feature transformation
        Parameters
        ----------
        X: 2d array of real numbers
            original data for all the features
        y: 1d array of real numbers
            original values of Quantity of Interest
        Notes
        ----
        For classical full PCE only the dimensions of X are needed.
        Values of X and y are used only for reduction procedure.
        """

        n_samples, n_features = check_array(X).shape
        self.n_input_features = n_features

        # Initialization of the combinations matrix
        self.combinations_mat = init_index_matrix(
            self.degree, n_features, self.dlist)
        self.n_output_features = self.combinations_mat.shape[1]

        # apply reduction if needed
        if self.reduction:
            # make reduction of the size of the matrix
            if self.basis_dim < self.n_output_features:
                XP = self.transform(X)
                self.combinations_mat = reduce_dimension(XP, y,
                                                         self.combinations_mat,
                                                         self.poly_type,
                                                         self.basis_dim,
                                                         self.single_iter_dim,
                                                         self.niter,
                                                         self.nrand,
                                                         self.clf)
                self.n_output_features = self.combinations_mat.shape[1]

            else:
                print('cannot apply reduction in this case')

        print('n_output_features: {}'.format(self.n_output_features))
        return self

    def transform(self, X, y=None):
        '''Transformation of features for given combinations matrix
        The output values are
        Parameters
        ----------
        X: 2d array of real numbers
            input features
        '''
        n_samples = check_array(X).shape[0]
        XP = np.empty((n_samples, self.n_output_features), dtype=X.dtype)

        for i in np.arange(self.n_output_features):
            XP[:, i] = 1  # np.ones(XP[:, i].shape)
            for j in np.arange(self.n_input_features):

                coef = np.zeros(1 + self.combinations_mat[j, i], dtype=np.float64)
                coef[-1] = 1.0

                if j >= len(self.poly_type):
                    poly_type = 'hermite_probabilistic'
                    print('WARNING!!! polynomial type not defined for index {}'.format(j))
                    print('Hermite probabilistic polynomials used by default ')
                else:
                    poly_type = self.poly_type[j]

                if poly_type == 'legendre':
                    poly_values = np.polynomial.legendre.legval(X[:, j], coef)
                elif poly_type == 'hermite_probabilistic':
                    poly_values = np.polynomial.hermite_e.hermeval(X[:, j], coef)
                elif poly_type == 'hermite_physicist':
                    poly_values = XP[:, i], np.polynomial.hermite.hermval(X[:, j], coef)
                elif poly_type == 'chebyshev':
                    poly_values = np.polynomial.chebyshev.chebval(X[:, j], coef)
                elif poly_type == 'laguerre':
                    poly_values = np.polynomial.laguerre.lagval(X[:, j], coef)
                else:
                    raise Exception('polynomial type not defined {}'.format(poly_type))

                XP[:, i] = np.multiply(XP[:, i], poly_values)

        return XP
