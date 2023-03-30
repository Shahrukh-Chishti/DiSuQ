from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import OrderingOptimization,PolynomialOptimization
from DiSuQ.Torch.optimization import lossAnharmonicity,lossTransitionFlatness
from torch import tensor
from numpy import arange,array
from DiSuQ.utils import plotCompare
from DevSuQ.utils import plotTrajectory
from time import perf_counter,sleep
from torch import set_num_threads
set_num_threads(64)

Ec,Ej = 100,20
gamma = 1.5
n = 2
array_range = arange(1,8)

path_array, path_approx = [],[]
for N in array_range:
    basis = [6]+[n]*N
    print('Array Range :',N)
    print(basis)
    
    start = perf_counter()
    circuit = models.fluxoniumArray(basis,N=N,Ec=Ec,Ej=Ej,sparse=True)
    H_LC = circuit.chargeHamiltonianLC()
    H_J = circuit.josephsonCharge
    print(H_LC.shape)
    end = perf_counter()
    print('Time Construction:',end-start)
    
    start = perf_counter()
    E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,dict(),grad=False)[:3]
    path_array.append((E1-E0,E2-E1))
    del circuit
    end = perf_counter()
    print('Time Diagonalization:',end-start)
    
    El = gamma*Ej/N
    basis = {'O':[1000],'I':[],'J':[]}
    start = perf_counter()
    circuit = models.fluxonium(basis,El,Ec,Ej,sparse=False)
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson
    print(H_LC.shape)
    E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,{'I':tensor(0.0)},grad=False)[:3]
    path_approx.append((E1-E0,E2-E1))
    del circuit
    end = perf_counter()
    print('Time:',end-start)
    print(N,'-------------')
    
path_array, path_approx = array(path_array),array(path_approx)

plotTrajectory(array_range,{'array':path_array,'approx':path_approx},'Fluxonium - Quasi approximation - Oscillator Basis','E10','E21',save=True)