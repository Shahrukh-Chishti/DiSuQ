from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import OrderingOptimization,PolynomialOptimization
from DiSuQ.Torch.optimization import loss_Transition
from torch import tensor,cuda,set_num_threads
from numpy import arange,linspace
from torch.linalg import eig as eigsolve,inv
import sys
import gc
from DiSuQ.utils import plotCompare
from DiSuQ.Torch.circuit import hamiltonianEnergy
import pickle
from time import perf_counter
set_num_threads(30)

flux_range = tensor(linspace(0,1,4))
flux_profile = [[flux] for flux in flux_range]
flux_point = ('I')


basis = {'O':[5],'I':[12,12],'J':[]}
circuit = models.shuntedQubit(basis,sparse=False)
#print(circuit.circuitComponents())
H_LC = circuit.kermanHamiltonianLC()
H_J = circuit.kermanHamiltonianJosephson({'I':.125})
H = H_LC+H_J
print(H.shape)
start = perf_counter()
print(hamiltonianEnergy(H)[:3])
print(perf_counter()-start)


H_J = circuit.kermanHamiltonianJosephson({'I':.05})
H = H_LC+H_J

start = perf_counter()
print(hamiltonianEnergy(H)[:3])
print(perf_counter()-start)

sys.exit(0)

basis = [1,1,1]
basis = [6,6,6]
basis = [4,4,4]

circuit = models.shuntedQubit(basis,sparse=False)
print(circuit.circuitComponents())

H_LC = circuit.chargeHamiltonianLC()
H_J = circuit.josephsonCharge({'I':tensor(.125)})

H = H_LC+H_J
print(H.shape)
#print(hamiltonianEnergy(H_J({})))
#eigenenergies,_ = eigsolve(H_J+H_LC)
#print(eigenenergies.real.sort()[0][:3])
start = perf_counter()
print(hamiltonianEnergy(H)[:3])
print(perf_counter()-start)

#print(H)

H_LC = circuit.chargeHamiltonianLC()
H_J = circuit.josephsonCharge({'I':tensor(.5)})
H = H_LC+H_J
print(H.shape)
#eigenenergies,_ = eigsolve(H_J+H_LC)
#print(eigenenergies.real.sort()[0][:3])
start = perf_counter()
print(hamiltonianEnergy(H)[:3])
print(perf_counter()-start)
