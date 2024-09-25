from DiSuQ.Torch.models import transmon
from DiSuQ.Torch.components import complex,float
from DiSuQ.Torch.discovery import uniformParameters,initializationParallelism,lossTransition,initializationSequential
from DiSuQ.Torch.optimization import GradientDescent
from DiSuQ.Torch.discovery import lossAnharmonicity,lossTransition
from DiSuQ.Torch.circuit import Charge
from torch import tensor
from numpy import arange
from DiSuQ.utils import plotCompare
from torch import set_num_threads
set_num_threads(30)

basis = [256]
circuit = transmon(Charge,basis,sparse=False)
print(circuit.circuitComponents())
flux_profile = [[]]

lossObjective = lossTransition(tensor([.5],dtype=float),tensor([.25],dtype=float))
optim = GradientDescent(circuit,circuit,flux_profile,lossObjective)
optim.initAlgo(lr=1.)
print(optim.optimizer)
dLogs,dParams,dCircuit = optim.optimization(iterations=1000)

plotCompare(dLogs.index,dLogs,'Optimizing Transmon','iteration')
plotCompare(dParams.index,dParams,None,"iteration","parameters")
plotCompare(dCircuit.index,dCircuit,None,"iteration","energy")