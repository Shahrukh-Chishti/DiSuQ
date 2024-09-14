from multiprocess import Pool
#from multiprocessing import Pool
from DiSuQ.Torch.optimization import uniformParameters,initializationParallelism,truncNormalParameters,initializationSequential
from DiSuQ.Torch import models
from DiSuQ.Torch.components import J0,L0,C0
from DiSuQ.Torch import optimization
from DiSuQ.Torch.optimization import lossTransition
from scipy.optimize import LinearConstraint
from torch import tensor, float32 as float
from numpy import arange,linspace,array,argsort,vstack,isnan
from DiSuQ.utils import plotCompare,plotOptimization,plotBox,plotHeatmap
import pickle
from torch import set_num_threads
set_num_threads(32)

with open('../C-shuntedProfiling/target_fluxqubit.p', 'rb') as content: target_info = pickle.load(content)
target_spectrum = target_info['spectrum']
E10 = target_spectrum[:,1] - target_spectrum[:,0]
E20 = target_spectrum[:,2] - target_spectrum[:,0]
target = {'E10':E10[[0,20,-1]],'E20':E20[[0,20,-1]]}


Algo = ["Adam2","lrBFGS",'Nelder-Mead','LBFGS']

lossFunction = lossTransition(tensor(target['E10'],dtype=float),tensor(target['E20'],dtype=float))

iterations = 5; n_scape = 50

flux_range = linspace(0,1,3,endpoint=True)
flux_profile = [{'I':flux} for flux in flux_range]

def benchmarking(optimizer,initials,subspace):
    Adam2,lrBFGS,NelMea,LBFGS = [],[],[],[]
    for index,parameter in enumerate(initials):
        
        
        # Nelder Mead
        print('Nelder Mead')
        optimizer.circuit.initialization(parameter)
        optimizer.parameters,optimizer.IDs = optimizer.circuitParameters(subspace)
        NelMea.append(optimizer.minimization(lossFunction,flux_profile,subspace=subspace,
                                             method='Nelder-Mead',options=dict(fatol=1e-6)))
        
#         Conjugate Gradient
#         print('Conjugate Gradient')
#         optimizer.circuit.initialization(parameter)
#         optimizer.parameters,optimizer.IDs = optimizer.circuitParameters(subspace)
#         CG.append(optimizer.minimization(lossFunction,flux_profile,subspace=subspace,
#         method='CG',tol=1e-8,maxiter=iterations))
        
#         # Wolfe BFGS
#         print('Wolfe BFGS')
#         optimizer.circuit.initialization(parameter)     
#         optimizer.parameters,optimizer.IDs = optimizer.circuitParameters(subspace)        
#         Wolfe.append(optimizer.optimizationLBFGS(lossFunction,flux_profile,iterations=iterations))
        
        # lr-LBFGS
        print('lr LBFGS')
        optimizer.circuit.initialization(parameter)
        optimizer.parameters,optimizer.IDs = optimizer.circuitParameters(subspace)
        lrBFGS.append(optimizer.optimizationLBFGS(lossFunction,flux_profile,iterations=iterations,lr=.1))
        
        
    return Adam2,lrBFGS,NelMea,LBFGS


basis = {'O':[8],'J':[6,6],'I':[]}; rep = 'K'
C12 = .5 ; C22 = 20. ; C11 = C22
JJ1 = 150; JJ3 = 150 ; JJ2 = 50
ind = 800
circuit = models.shuntedQubit(basis,josephson=[JJ1,JJ2,JJ3],cap=[C11,C12,C22],ind=ind,sparse=False,symmetry=True)
static = circuit.circuitState()

optimizer = optimization.OrderingOptimization(circuit,representation=rep)

subspace = ['JJ1','JJ2','C1','C2'] ; N = 2
initials = uniformParameters(circuit,subspace,N)
#initials = truncNormalParameters(circuit,subspace,N,var=5)
len(initials)

OptimizationD = benchmarking(optimizer,initials[8:],subspace)
Adam2,lrBFGS,NelMea,LBFGS = OptimizationD