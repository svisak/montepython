import numpy as np
from scipy.optimize import minimize
from .utils import print_vector, print_matrix

def tune_mass_matrix(bayes, startpos, ndof=1, verbose=False, return_optimum=False, return_gradient=False, fmt='.3e'):
    '''
    Find a mass matrix using by optimizing the posterior with the BFGS algorithm
    and computing the approximate inverse Hessian at the optimum.

    :param bayes:
        A montepython.Bayes or montepython.SimpleBayes object.

    :param startpos:
        The start position vector.

    :param ndof: (optional)
        Scale (i.e., divide) the log posterior by this value during printing.
        Has no influence on the minimization itself, only used for printing.
        Default: 1

    :param verbose: (optional)
        Print information about the procedure.
        Default: False

    :param return_optimum: (optional)
        If True, the optimal parameter values found during minimization is returned
        alongside the mass matrix.
        Default: False

    :param return_gradient: (optional)
        If True, the gradient at the optimum is returned alongside the mass matrix.
        Default: False

    '''
    if verbose:
        print('Finding a mass matrix using BFGS minimization.')
        print(f'Start position: {startpos}')
    settings = {}
    settings['method'] = 'BFGS'
    settings['jac'] = True
    args = (bayes, ndof, verbose, fmt)
    result = minimize(minimization_cost_function, startpos, args=args, **settings)
    optimum = result.x
    hess_inv = result.hess_inv
    mass_matrix = np.linalg.inv(hess_inv)

    cost, grad = minimization_cost_function(optimum, bayes, ndof=ndof, verbose=False)
    return_value = mass_matrix
    msg = 'Return value: mass_matrix'
    if return_optimum and not return_gradient:
        return_value = (mass_matrix, optimum)
        msg = 'Return value: (mass_matrix, optimum)'
    elif not return_optimum and return_gradient:
        return_value = (mass_matrix, grad)
        msg = 'Return value: (mass_matrix, gradient)'
    elif return_optimum and return_gradient:
        return_value = (mass_matrix, optimum, grad)
        msg = 'Return value: (mass_matrix, optimum, gradient)'

    if verbose:
        print(result.message, end='\n')
        print(f'Cost function at optimum: {cost:{fmt}}')
        if return_optimum:
            print('MAP:')
            print_vector(optimum, fmt)
        if return_gradient:
            print('Gradient at MAP:')
            print_vector(grad, fmt)
        print(msg)

    return return_value

def minimization_cost_function(position, bayes, ndof=1, verbose=False, fmt='.3e'):
    bayes.evaluate(position)
    cost = -bayes.get_lnposterior_value()
    grad = bayes.get_nlp_gradient_value()
    if verbose:
        print(f'{cost / ndof:{fmt}}: cost function value at position:')
        print(f'{position}')
        print()
    return (cost, grad)
