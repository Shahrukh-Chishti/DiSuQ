from DiSuQ.Torch import models
from torch import tensor,set_num_threads
from numpy import linspace
import sys
from DiSuQ.Torch.circuit import hamiltonianEnergy,Kerman
from time import perf_counter
set_num_threads(30)

flux_range = tensor(linspace(0,1,4))
flux_profile = [[flux] for flux in flux_range]
flux_point = ('I')


basis = {'O':[5],'J':[12,12],'I':[]}
circuit = models.shuntedQubit(Kerman,basis,sparse=False)
#print(circuit.circuitComponents())
H_LC = circuit.hamiltonianLC()
H_J = circuit.hamiltonianJosephson({'I':.125})
H = H_LC+H_J
print(H.shape)
start = perf_counter()
print(hamiltonianEnergy(H)[:3])
print(perf_counter()-start)


H_J = circuit.hamiltonianJosephson({'I':.05})
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
