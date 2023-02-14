from multiprocess import Pool
#from multiprocessing import Pool
from DiSuQ.Torch.optimization import uniformParameters,initializationParallelism
from DiSuQ.Torch import models
from DiSuQ.Torch.optimization import OrderingOptimization,PolynomialOptimization
from DiSuQ.Torch.optimization import loss_Transition
from torch import tensor, float32 as float
from numpy import arange,linspace
from DiSuQ.utils import plotCompare
import pickle
from torch import set_num_threads
set_num_threads(30)


with open('../Examples/target_fluxqubit.p', 'rb') as content: target_info = pickle.load(content)
target_spectrum = target_info['spectrum']
E10 = target_spectrum[:,1] - target_spectrum[:,0]
E20 = target_spectrum[:,2] - target_spectrum[:,0]
target = {'E10':E10[[0,20,-1]],'E20':E20[[0,20,-1]]}


basis = {'O':[4],'I':[8,8],'J':[]}; rep = 'K'
circuit = models.shuntedQubit(basis,sparse=False)


flux_range = linspace(0,1,3,endpoint=True)


flux_profile = [{'I':flux} for flux in flux_range]


lossObjective = loss_Transition(tensor(target['E10'],dtype=float),tensor(target['E20'],dtype=float))


optim = OrderingOptimization(circuit,representation=rep)


parameters = uniformParameters(circuit,3)


with Pool(10) as multi:
    Search = multi.map(initializationParallelism(optim,lossObjective,flux_profile),parameters)
    
import ipdb;ipdb.set_trace()