from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import OrderingOptimization,PolynomialOptimization
from DiSuQ.Torch.optimization import loss_Anharmonicity,lossTransitionFlatness
from torch import tensor
from numpy import arange,array
from DiSuQ.utils import plotCompare
from DevSuQ.utils import plotTrajectory
from torch import set_num_threads
set_num_threads(64)

Ec,Ej = 100,20
gamma = 1.5
n = 2
array_range = arange(3,7)

path_array, path_approx = [],[]
for N in array_range:
    basis = [8]+[n]*N
    circuit = models.fluxoniumArray(basis,N=N,Ec=Ec,Ej=Ej,sparse=False)
    H_LC = circuit.chargeHamiltonianLC()
    H_J = circuit.josephsonCharge
    E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,dict()).detach().numpy()[:3]
    path_array.append((E1-E0,E2-E1))
    del circuit
    
    El = gamma*Ej/N
    basis = {'O':[1000],'I':[],'J':[]}
    circuit = models.fluxonium(basis,El,Ec,Ej,sparse=False)
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson
    E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,{'I':tensor(0.0)}).detach().numpy()[:3]
    path_approx.append((E1-E0,E2-E1))
    del circuit
    
    print(N,'-------------')
    
path_array, path_approx = array(path_array),array(path_approx)

plotTrajectory(array_range,{'array':path_array,'approx':path_approx},'Fluxonium - Quasi approximation','E10','E21',save=True)