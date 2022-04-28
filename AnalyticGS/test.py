import numpy,sys
from ground_state_search import scipy_ground_state,polynomial,solve_smallest_root,gradient,squarePolynomial
from scipy.optimize import minimize as sp_minimize
import matplotlib.pyplot as plt
# highest power ----> constant
quintic = [1,40,0,-2,12,10]
quartic = [1,21,5,-2,10]
cubic = [2,1,4,3]
quadratic = [1,19,3]
generic = [2,23,45,23,1,41,36,16,13,61,61,3,8,6,9,6]

poly = generic
x0 = numpy.array([-50])

roots = numpy.roots(poly)
print('roots',roots)
print('smallest root',min(roots))

f = polynomial(poly)
loss = lambda x:f(x)*f(x)
poly2 = squarePolynomial(poly)
poly2 = numpy.array(poly2)
poly2 = numpy.flip(poly2)

#print('f(root)',f(min(roots)))
#print('polyval(roots)',numpy.polyval(poly,min(roots)))
iterations = 5
x,X = solve_smallest_root(poly2,x0,iterations)
print('optimised solution:',x)
plt.plot(range(iterations),X)
#plt.plot()
plt.xlabel('iteration')
plt.ylabel('solution')
plt.title('Super-Newton root minima');plt.show()

sys.exit(0)

I = gradient(poly)
res = sp_minimize(loss,x0,method='Newton-CG',jac=I)
print(res)

iteration_set = numpy.arange(3,25,dtype=int)
starting_set = numpy.arange(-30,-25,.1)
ground_eval = scipy_ground_state(poly,iteration_set,starting_set)
error = abs(ground_eval-min(roots))/abs(min(roots))*100

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
