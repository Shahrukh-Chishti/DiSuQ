import numpy
from ground_state_search import scipy_ground_state,polynomial,solve_smallest_root,gradient
from scipy.optimize import minimize as sp_minimize
import matplotlib.pyplot as plt

quintic = [1,40,0,-2,12,10]
cubic = [2,1,4,3]
generic = [2,23,45,23,1,41,36,16,13,61,61,3,8,6,9,6]
x0 = [-1]
x0 = numpy.array(x0)

roots = numpy.roots(generic)
print('roots',roots)
print('smallest root',min(roots))

f = polynomial(generic)
loss = lambda x:f(x)*f(x)

print('f(root)',f(min(roots)))
print('polyval(roots)',numpy.polyval(generic,min(roots)))
#x = solve_smallest_root(cubic,x0,50)
#print('optimised solution:',x)

I = gradient(generic)
res = sp_minimize(loss,x0,method='Newton-CG',jac=I)
print(res)

iteration_set = numpy.arange(3,10,dtype=int)
starting_set = numpy.arange(-15,-5,.5)
ground_eval = scipy_ground_state(generic,iteration_set,starting_set)
error = abs(ground_eval-min(roots))

## plot dynamics for iterative algorithm
fig,ax = plt.subplots(nrows=1,ncols=2)

bp = ax[0].boxplot(error)
ax[0].set_xticklabels(starting_set)
ax[0].set_xlabel('starting_set')

bp = ax[1].boxplot(error.transpose())
ax[1].set_xticklabels(iteration_set)
ax[1].set_xlabel('iteration')

plt.ylabel('error')
plt.show()

import ipdb;ipdb.set_trace()
