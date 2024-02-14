from DiSuQ.Torch import models as TorchModels
from DiSuQ.Torch.optimization import OrderingOptimization
from DiSuQ.Torch.optimization import lossAnharmonicity,lossTransitionFlatness

from DiSuQ.Kerman import models as KermanModels
from scipy.optimize import minimize

from torch import tensor
from numpy import arange,linspace,var,logspace
from DiSuQ.utils import plotCompare
from torch import set_num_threads
from time import perf_counter
from memory_profiler import memory_usage
set_num_threads(32)

n_iter = 50
n_bases = 25
n_flux = 32

def torchProfiling(N,El=10.,Ec=100.,Ej=50):
    basis = {'O':[N],'I':[],'J':[]}; rep = 'K'
    circuit = TorchModels.fluxonium(basis,El=El,Ec=Ec,Ej=Ej,sparse=False)
    flux_range = tensor(linspace(0,1,n_flux,endpoint=True))
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson
    H_J = H_J({'I':tensor(0.0)})
    #flux_range = [[flux] for flux in flux_range]
    #E0,(E1,E2) = circuit.spectrumManifold('I',flux_range,H_LC,H_J,[1,2])
        
def numpyProfiling(N,El=10.,Ec=100.,Ej=50):
    basis = {'O':[N],'I':[],'J':[]}; rep = 'K'
    circuit = KermanModels.fluxonium(basis,El=El,Ec=Ec,Ej=Ej)
    flux_range = linspace(0,1,n_flux,endpoint=True)
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson
    H_J = H_J({'I':0.0})
    #flux_range = [[flux] for flux in flux_range]
    #E0,(E1,E2) = circuit.spectrumManifold('I',flux_range,H_LC,H_J,[1,2])
    
if __name__ == "__main__":
    #numpyProfiling(50)
    #import sys;sys.exit(0)
    torch_comp,torch_mem = [],[]
    numpy_comp,numpy_mem = [],[]
    bases = logspace(1,3,n_bases,dtype=int)
    print(bases)
    arguments = {'El':10.,'Ec':100.,'Ej':50.}
    for N in bases:
        print(N)
        start = perf_counter()
        mem = memory_usage((torchProfiling,(N,)))
        torch_comp.append(perf_counter()-start)
        torch_mem.append(max(mem))
        
        start = perf_counter()
        mem = memory_usage((numpyProfiling,(N,)))
        numpy_comp.append(perf_counter()-start)
        numpy_mem.append(max(mem))
        
    print(torch_comp,numpy_comp)
    print(torch_mem,numpy_mem)

    plotCompare(bases,{'torch':torch_mem,'numpy':numpy_mem},'Instantiation Memory Profile','basis','peak memory',
                export='pdf',size=(600,600))
    plotCompare(bases,{'torch':torch_comp,'numpy':numpy_comp},'Instantiation Computation Profile','basis','time',
                export='pdf',size=(600,600))