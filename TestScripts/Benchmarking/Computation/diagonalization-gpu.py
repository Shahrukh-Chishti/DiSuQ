from DiSuQ.Torch import models as TorchModels
from DiSuQ.Kerman import models as KermanModels

from torch import tensor
from numpy import arange,linspace,var,logspace
from DiSuQ.utils import plotCompare
from time import perf_counter
import memory_profiler as mp
import pandas

import torch,numpy
from threadpoolctl import threadpool_limits
from DiSuQ.utils import plotCompare


def torchInstantiationFluxonium(N,El=10.,Ec=100.,Ej=50.,sparse=False):
    basis = {'O':[N],'I':[],'J':[]}; rep = 'K'
    circuit = TorchModels.fluxonium(basis,El=El,Ec=Ec,Ej=Ej,sparse=sparse)
    flux_point = {'I':tensor(0.)}
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson(flux_point)
    return H_LC+H_J

def torchInstantiationTransmon(N,El=10.,Ec=100.,Ej=50.,sparse=False):
    basis = {'O':[N],'I':[],'J':[]}; rep = 'Q'; basis = [N]
    circuit = TorchModels.transmon(basis,Ec=Ec,Ej=Ej,sparse=sparse)
    flux_point = {'I':tensor(0.)}
    H_LC = circuit.chargeHamiltonianLC()
    H_J = circuit.josephsonCharge(flux_point)
    return H_LC+H_J

def numpyInstantiation(N,El=10.,Ec=100.,Ej=50,sparse=False):
    basis = {'O':[N],'I':[],'J':[]}; rep = 'K'
    circuit = KermanModels.fluxonium(basis,El=El,Ec=Ec,Ej=Ej)
    flux_point = {'I':0.}
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson(flux_point)
    return H_LC+H_J


threads = [16,32,40]
n_bases = 7
device = torch.device(0)
#import ipdb;ipdb.set_trace()

dSpa = []
vectors = False
bases = logspace(2,3.2,n_bases,dtype=int)
print(bases)
arguments = {'El':10.,'Ec':100.,'Ej':50.}
for n_threads in threads:
    print(n_threads,'----------')
    for N in bases:
        print(N,'==============')
        torch.set_num_threads(n_threads)
        
        H = torchInstantiationTransmon(N,sparse=False).to(float)
        print('dense-diagonalization')
        start = perf_counter()
        mem = mp.memory_usage((torch.lobpcg,(H.to(device),4)))
        time = perf_counter()-start
        dSpa.append({'lib':'torch','stage':'diag','hilbert':N,'threads':n_threads,
                        'peak_memory':max(mem)-min(mem),'time':time,'vectors':vectors,'rep':'dense'})
                              
        H = torchInstantiationTransmon(N,sparse=True).to(float)
        print('sparse-diagonalization')
        start = perf_counter()
        mem = mp.memory_usage((torch.lobpcg,(H.to(device),4)))
        time = perf_counter()-start
        dSpa.append({'lib':'torch','stage':'diag','hilbert':N,'threads':n_threads,
                        'peak_memory':max(mem)-min(mem),'time':time,'vectors':vectors,'rep':'sparse'})
                              
        
dSpa = pandas.DataFrame(dSpa)
dSpa.to_csv('sparse-dense-gpu-re.csv')