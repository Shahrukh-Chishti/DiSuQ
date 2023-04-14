from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import OrderingOptimization,PolynomialOptimization
from DiSuQ.Torch.optimization import lossAnharmonicity,lossTransitionFlatness
from torch import tensor
from numpy import arange,array
from DiSuQ.utils import plotCompare
from DiSuQ.utils import plotTrajectory
from time import perf_counter,sleep
from torch import set_num_threads
set_num_threads(32)


def execute(gamma):
    Ec,Ej = 1.5,45.
    n = 3
    N_array = 14
    array_range = arange(1,N_array)
    n_basis = 512*2

    Basis = dict()
    Basis['14'] = [1,1,1,1,1,1,1,1,1,1,1,1,2,2]
    Basis['13'] = [1,1,1,1,1,1,1,1,1,1,2,2,2]
    Basis['12'] = [1,1,1,1,1,1,1,2,2,2,2,2]
    Basis['11'] = [1,1,1,1,1,2,2,2,2,2,4]
    Basis['10'] = [1,1,1,2,2,2,2,2,3,4]
    Basis['9'] = [1,1,1,2,2,3,3,4,6]
    Basis['8'] = [1,2,2,3,3,3,5,6]
    Basis['7'] = [2,3,3,3,4,6,8]
    Basis['6'] = [3,3,5,7,9,12]
    Basis['5'] = [7,8,9,12,15]
    Basis['4'] = [12,16,20,25]
    Basis['3'] = [50,75,100]
    Basis['2'] = [500,1000]


    path_array, path_approx, path_cosine = [],[],[]


    for N in array_range:
        # N= N+1 ; basis = arange(N,0,-1,int).tolist()  # Linear
        # basis = [8]+[n]*N # static
        basis = Basis[str(N+1)]
        basis.reverse()
        assert len(basis) == N+1
        print('Array Range :',N)
        print(basis)

        start = perf_counter()
        circuit = models.fluxoniumArray(basis,gamma=gamma,N=N,Ec=Ec,Ej=Ej,sparse=True)
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

    #path_array = array(path_array)
    #plotTrajectory(array_range,{'approx':path_array},'Quasi approximation-Charge Basis-Gamma'+str(gamma),'E10','E21',save=True)
    #import sys;sys.exit(0)

    N_approx = 25
    array_range = arange(1,N_approx)
    print('Approximate Cosine Model')
    for N in array_range:

        El = gamma*Ej/N
        basis = [n_basis]
        start = perf_counter()
        circuit = models.transmon(basis,Ej,Ec,sparse=False)
        H_LC = circuit.chargeHamiltonianLC()
        H_LC -= N*gamma*Ej* circuit.backend.displacementCharge(n_basis,1/N)/2
        H_LC -= N*gamma*Ej* circuit.backend.displacementCharge(n_basis,-1/N)/2
        H_J = circuit.josephsonCharge
        print(H_LC.shape)
        E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,{'I':tensor(0.0)},grad=False)[:3]

        path_approx.append((E1-E0,E2-E1))
        del circuit
        end = perf_counter()
        print('Time:',end-start)
        print(N,'-------------')


    N_approx = 25
    array_range = arange(1,N_approx)
    n_basis *= 2
    print('Expansion Cosine Model')
    for N in array_range:
        El = gamma*Ej/N
        basis = [n_basis]
        #basis = {'O':[n_basis],'I':[],'J':[]}    
        start = perf_counter()
        print(El,Ec,Ej)
        circuit = models.fluxonium(basis,El,Ec,Ej,sparse=False)
        H_LC = circuit.chargeHamiltonianLC()
        #H_LC = circuit.kermanHamiltonianLC()
        H_J = circuit.josephsonCharge
        #H_J = circuit.kermanHamiltonianJosephson
        print(H_LC.shape)
        E0,E1,E2 = circuit.circuitEnergy(H_LC,H_J,{'I':tensor(0.0)},grad=False)[:3]
        path_cosine.append((E1-E0,E2-E1))
        del circuit
        end = perf_counter()
        print('Time:',end-start)
        print(N,'-------------')

    path_array, path_approx, path_cosine = array(path_array),array(path_approx),array(path_cosine)

    plotTrajectory(array_range,{'array':path_array,'approx':path_approx,'cosine':path_cosine},'Quasi approximation-Charge Basis-Gamma'+str(gamma),'E10','E21',html=True,export='pdf',size=(600,600))

execute(gamma = 2.)
execute(gamma = 1.)
execute(gamma = 10.)