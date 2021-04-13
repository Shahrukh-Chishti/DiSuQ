import pickle,numpy
from qircuit_shunted_qubit import *
from utils import designRandomCircuit
import matplotlib.pyplot as plt

with open('target_fluxqubit.p', 'rb') as content: target_info = pickle.load(content)
target_spectrum = target_info['spectrum']

#phiExt = numpy.linspace(0,1,41,endpoint=True)

N = 6
c_specs        = {'dimension': 6, 'low': 1.,  'high': 100.,  'keep_prob': 0.5}
j_specs        = {'dimension': 6, 'low': 99., 'high': 1982., 'keep_num':  3}
l_specs        = {'dimension': 6, 'low': 75., 'high': 300.,  'keep_prob': 0.5}
circuit_bounds = {'C':{'low':c_specs['low'],'high':c_specs['high']},
                  'J':{'low':j_specs['low'],'high':j_specs['high']},
                  'L':{'low':l_specs['low'],'high':l_specs['high']}}

# initialise circuit
random_circuit = designRandomCircuit(c_specs,j_specs)
init_pos = numpy.concatenate([random_circuit['capacities'],random_circuit['junctions'],random_circuit['inductances']])
# solve circuit
loss = lossBuilder(target_spectrum,phiExt_sweep,circuit_bounds)
bounds = defineComponentBounds(circuit_bounds,N)

optimizer    = ps.single.GlobalBestPSO(n_particles = n_particles, dimensions = len(sim_info_dict['x_init_squeezed']),options = self.social_options, bounds = sim_info_dict['bounds_collection'], init_pos = init_pos_mod)

# optimised circuit
C,J = circuitUnwrap(res.x)
Cmat,Jmat = arrayMatrix(C),arrayMatrix(J)
