from multiprocess import Pool
from DiSuQ.Torch.discovery import uniformParameters,initializationParallelism,lossTransition
from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import GradientDescent
from DiSuQ.Torch.circuit import Kerman
from torch import tensor, float32 as float
from numpy import linspace
import pickle
from torch import set_num_threads
set_num_threads(6)


with open('/Users/chishti/DiSuQ/Examples/C-shuntedProfiling/target_fluxqubit.p', 'rb') as content:
    target_info = pickle.load(content)
target_spectrum = target_info['spectrum']
print(target_spectrum.shape)
E10 = target_spectrum[:,1] - target_spectrum[:,0]
E21 = target_spectrum[:,2] - target_spectrum[:,1]
target = {'E10':E10[[0,20,-1]],'E21':E21[[0,20,-1]]}


basis = {'O':[2],'J':[3,3],'I':[]}
circuit = models.shuntedQubit(Kerman,basis,sparse=False)

flux_range = linspace(0,1,3,endpoint=True)
flux_range = tensor(flux_range)
flux_profile = [[tensor(flux)] for flux in flux_range]

lossObjective = lossTransition(tensor(target['E10'],dtype=float),tensor(target['E21'],dtype=float))
optim = GradientDescent(circuit,circuit,flux_profile,lossObjective)
subspace = [component.ID for component in circuit.network]
parameters = uniformParameters(circuit,subspace,10,5)

parallel = initializationParallelism(optim)
print('test run-------------pass')
with Pool(10) as multi:
    Search = multi.map(initializationParallelism(optim,iterations=5),parameters)
    
import ipdb;ipdb.set_trace()