import torch,numpy
from torch.autograd import Variable
from scipy.optimize import minimize as sp_minimize


def polynomial(poly):
    N = len(poly)-1
    def evaluate(x):
        func = 0
        for index,coeff in enumerate(poly):
            power = N-index
            func += coeff*(x**(power))
        return func
    return evaluate

def gradient(poly):
    N = len(poly)-1
    f = polynomial(poly)
    def Ist(x):
        func = 0
        for index,coeff in enumerate(poly[:-1]):
            power = N-index
            func += power*coeff*(x**(power-1))
        return 2*func*f(x)
    return Ist

def curvature(poly):
    N = len(poly)-1
    f = polynomial(poly)
    I = gradient(poly)
    def IInd(x):
        func = 0
        for index,coeff in enumerate(poly[:-2]):
            power = N-index
            func += (power)*(power-1)*coeff*(x**(power-2))
        return 2*func*f(x) + 2*(I(x)**2)
    return IInd

def next(x,I,II):
    update = I(x)/II(x)
    return x-update

def solve_smallest_root(poly,x,iterations):
    f = polynomial(poly)
    I = gradient(poly)
    II = curvature(poly)
    for iteration in range(iterations):
        print(x,I(x),II(x))
        x = next(x,f,I)
    return x

def scipy_ground_state(poly,iteration_set,starting_set):
    f = polynomial(poly)
    I = gradient(poly)
    loss = lambda x:f(x)*f(x)
    ground_solve = numpy.ones((len(iteration_set),len(starting_set)))
    for i,iteration in enumerate(iteration_set):
        for j,starting in enumerate(starting_set):
            x0 = numpy.array([starting])
            res = sp_minimize(loss,x0,method='Newton-CG',jac=I,options={'maxiter':iteration})
            ground_solve[i,j] = res.x[0]
    return ground_solve

