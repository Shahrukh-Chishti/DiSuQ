import pickle,numpy
from qircuit_shunted_qubit import *
from utils import designRandomCircuit
import matplotlib.pyplot as plt
import pyswarms as ps

N = 6
c_specs        = {'dimension': N, 'low': 1.,  'high': 100.,  'keep_prob': 0.5}
j_specs        = {'dimension': N, 'low': 99., 'high': 1982., 'keep_num':  3}
l_specs        = {'dimension': N, 'low': 75., 'high': 300.,  'keep_prob': 0.5}
circuit_bounds = {'C':{'low':c_specs['low'],'high':c_specs['high']},
                  'J':{'low':j_specs['low'],'high':j_specs['high']},
                  'L':{'low':l_specs['low'],'high':l_specs['high']}}
merit_options = {'max_peak': 1.5, 'max_split': 10, 'norm_p': 4, 'flux_sens': True, 'max_merit': 100}

# initialise circuit
random_circuit = designRandomCircuit(c_specs,j_specs)
init_pos = numpy.concatenate([random_circuit['capacities'],random_circuit['junctions'],random_circuit['inductances']])
# solve circuit
bounds = defineComponentBounds(circuit_bounds,N)

n_particles = 100
social_options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer    = ps.single.GlobalBestPSO(n_particles = n_particles, dimensions = N,
                                       options = social_options, bounds = bounds, init_pos = init_pos)
cost,pos = optimizer.optimize(merit_DoubleWell, merit_options = merit_options,
                             iters = max_iter, verbose = 1, print_step = 1)

# optimised circuit
C,J = circuitUnwrap(res.x)
Cmat,Jmat = arrayMatrix(C),arrayMatrix(J)
