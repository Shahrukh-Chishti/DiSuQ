## 2 node solver
## L-BFGS

import pickle,numpy
from qircuit_shunted_qubit import *
from utils import designRandomCircuit
import matplotlib.pyplot as plt

with open('target_fluxqubit.p', 'rb') as content: target_info = pickle.load(content)
target_spectrum = target_info['spectrum']

phiExt = numpy.linspace(0,1,41,endpoint=True)
N = 3
c_specs = {'dimension': 3, 'low': 0., 'high': 10000, 'keep_prob': 1.}
j_specs = {'dimension': 3, 'low': 0., 'high': 200, 'keep_num':  3}
circuit_bounds = {'C':{'low':c_specs['low'],'high':c_specs['high']},
                  'J':{'low':j_specs['low'],'high':j_specs['high']}}

# initialise circuit
random_circuit = designRandomCircuit(c_specs,j_specs)
init_pos = numpy.concatenate([random_circuit['capacities'],random_circuit['junctions']])
Cmat,Jmat = arrayMatrix(init_pos[:3]),arrayMatrix(init_pos[3:])
"""
# solve circuit
loss = lossBuilder(target_spectrum,phiExt,circuit_bounds)
bounds = defineComponentBounds(circuit_bounds,N)
res = sp_minimize(loss, init_pos, method = 'L-BFGS-B', bounds = bounds, options = {'maxiter': 100})
# optimised circuit
C,J = circuitUnwrap(res.x)
Cmat,Jmat = arrayMatrix(C),arrayMatrix(J)

print(c_specs)
print(j_specs)
print('optimisation result:')
print(res)
print('\n\n')
print('Capacitances[C11,C12,C22]:',C)
print('Josephsons[J11,J12,J22]:',J)
"""
## plot Eigenspectrum
spec = [eigen2Node(Cmat,Jmat,phi) for phi in phiExt]
spec = numpy.array(spec)
E0 = numpy.array([spec[i][0] for i in range(len(spec))])
spec = (spec.T - E0).T

fig, axes = plt.subplots(figsize=(12,6))
axes.plot(phiExt, spec[:,1], label='ground state')
axes.plot(phiExt, spec[:,2], label='Ist excited')
axes.legend()
plt.xlabel('$\phi_{ext} / \phi_0$')
plt.ylabel('Energy')
plt.rcParams.update({'font.size': 13})
plt.show()
