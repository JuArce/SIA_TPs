import math
import numpy as np
from scipy.optimize import minimize
from autograd.misc.optimizers import adam, sgd
import numdifftools as nd
from datetime import datetime

values = [[(4.4793, -4.0765, -4.0765), 0], [(-4.1793, -4.9218, 1.7664), 1], [(-3.9429, -0.7689, 4.883), 1]]


def g(val):
    try:
        return math.exp(val) / (1 + math.exp(val))
    except:
        return 1


def F(W, w, w0, epsilon):
    external_sum = 0
    for i in range(0, 2):
        internal_sum = 0
        for j in range(0, 3):
            internal_sum += w[i][j] * epsilon[j]
        internal_sum -= w0[i]
        external_sum += W[i + 1] * g(internal_sum)
    return g(external_sum - W[0])


def E(x, i=None, u=None):
    W = x[0:3]
    w = [x[3:6], x[6:9]]
    w_0 = x[9:11]
    return sum((OUT - F(W, w, w_0, IN)) ** 2 for (IN, OUT) in values)


def print_result(x, time, method):
    print("****************************************************")
    print(method)
    print("W = " + str(x[0:3]))
    print("w = " + str(x[3:6]) + "\n\t " + str(x[6:9]))
    print("w0 = " + str(x[9:11]))
    print("Error: " + str(E(x)))
    print("time elapsed: " + str(time))
    print("****************************************************\n")



def __main__():
    x = np.zeros(11)

    # Método del gradiente descendiente: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb
    time = datetime.now()
    result = minimize(E, x, method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None,
                      options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08,
                               'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20,
                               'finite_diff_rel_step': None})
    # result = sgd(nd.Gradient(E), x, step_size=0.1, num_iters=1500)
    time = datetime.now() - time
    print_result(result.x, time, 'Método del gradiente descendiente')

    # Metodo del gradiente conjugado: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html#optimize-minimize-cg
    time = datetime.now()
    result = minimize(E, x, method='CG', jac=None, tol=None, callback=None,
                      options={'gtol': 1e-05, 'norm': math.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None,
                               'disp': False, 'return_all': False, 'finite_diff_rel_step': None})
    time = datetime.now() - time
    print_result(result.x, time, 'Método del gradiente conjugado')



    # Método de adam: https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py
    time = datetime.now()
    result = adam(nd.Gradient(E), x, step_size=0.99)
    time = datetime.now() - time
    print_result(result, time, 'Método de Adam')


if __name__ == "__main__":
    __main__()
