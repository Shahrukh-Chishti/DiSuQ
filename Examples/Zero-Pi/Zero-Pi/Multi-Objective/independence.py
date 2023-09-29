from DiSuQ.Torch import models
from torch import tensor,stack
from numpy import arange,linspace,meshgrid,array,log,argsort,pi,std,logspace,log10,stack,concatenate
from DiSuQ.utils import plotCompare,plotHeatmap,plotBox,plotScatter
from DiSuQ.Torch.optimization import uniformParameters,truncNormalParameters,initializationSequential
from DiSuQ.Torch import optimization
from DiSuQ.Torch.optimization import lossDegeneracyTarget,lossDegeneracyWeighted,domainParameters
from DiSuQ.Torch.components import indE,capE
from torch import set_num_threads
import pickle
set_num_threads(64)


pairs = {'Lx':'Ly','Cx':'Cy','Jx':'Jy','CJx':'CJy'}

flux_profile = [{'Lx':tensor(.5)},{'Lx':tensor(.45)}]

def optimizationAnalysis(init,subspace,Search,success=1.):
    Loss,Success,Paths = [],[],[]
    for index,(init,(dLogs,dParams,dCircuit)) in enumerate(zip(init,Search)):
        if len(dLogs) > 0:
            Paths.append(dCircuit[subspace].to_numpy())
            loss = dLogs['loss'].to_numpy()
            Loss.append(loss[-1])
            if loss[-1] < success:
                Success.append(len(loss))
    return Paths,Loss,Success

def analysisPlotting(Optimization,Algo=['LBFGS']):
    paths = dict()
    losse = dict()
    for algo,(Paths,Loss,Success) in zip(Algo,Optimization):
        indices = argsort(Loss)[0]
        paths[algo] = Paths[indices]
        losse[algo] = Loss
    return paths,losse

def lossScapeBounds(paths):
    Paths = []
    for algo,path in paths.items():
        Paths.append(path)
    Paths = vstack(Paths)
    Paths = Paths[:, ~isnan(Paths).any(axis=0)]
    return Paths.min(0),Paths.max(0)

E0 = 30
El = .0022 * E0
EcS = .0058 * E0
Ej = E0
EcJ = E0

Ep = .1
El = Ep/990.
Ej = Ep/8.3
EcS = Ep/10000.
EcJ = Ep*Ep/Ej/8

Ec = 1/(1/EcS-1/EcJ)

print(El,Ec,EcJ,Ej)

L0 = indE(1e-7); L_ = indE(5e-2); print('Inductance Bound(GHz):',L_,L0+L_)
C0 = capE(1e-16); C_ = capE(1e-10) ; print('Capacitance Bound(GHz):',C_,C0+C_)
CJ0 = capE(1e-16); CJ_ = capE(1e-10) ; print('Shunt Bound(GHz):',CJ_,CJ0+CJ_)
J0 = 250. ; J_ = 1e-3 ; print('Junction Bound(GHz):',J_,J0+J_)
# components['Jx'].J0 = J0 ; components['Jy'].J0 = J0


basis = {'Chi':7,'Theta':12,'Phi':30}
rep = 'R'; flux_point = ['Lx','Ly']
circuit = models.zeroPi(basis,Ej=10.,Ec=5.,El=10.,EcJ=50.,sparse=False,
                        symmetry=True,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(CJ_,CJ0),
                        ridge=True,flux0=24*pi)
static = circuit.circuitState()


N = 2; subspace = ['Lx','Cx','Jx','CJx']
initials = uniformParameters(circuit,subspace,N)

Lx = array([.5])
Cx = array([5e-3,.5,10,100.])
Jx = array([.5,10,100,200.])
CJx = array([5e-3,.5,10,100.])

initials = domainParameters([Lx,Cx,Jx,CJx],circuit,subspace)

for init in initials:    
    El,Ej,EcJ,Ec = init['Lx'],init['Jx'],init['CJx'],init['Cx']
    #Ec = Ep**2/8/Ej
    print(EcJ/Ec,Ej/Ec,'\t',Ej/El,EcJ/El)
    
optimizer = optimization.OrderingOptimization(circuit,representation=rep)

D0 = 1; delta0 = 0
lossFunction = lossDegeneracyWeighted(delta0,D0)

LBFGS_D = []
for index,parameter in enumerate(initials):
    print(optimizer.circuitState())
    print(index,parameter)
    optimizer.circuit.initialization(parameter)  
    optimizer.parameters,optimizer.IDs = optimizer.circuitParameters()
    LBFGS_D.append(optimizer.minimization(lossFunction,flux_profile,
                    method='L-BFGS-B',subspace=(),options=dict(ftol=0,maxiter=15)))
    
    
Result = [optimizationAnalysis(initials,subspace,LBFGS_D)]
paths_D,losse_D = analysisPlotting(Result)

with open('ind_Degeneracy.pickle', 'wb') as handle:
    pickle.dump(LBFGS_D, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
D0 = 0; delta0 = 1
lossFunction = lossDegeneracyWeighted(delta0,D0)

LBFGS_Delta = []
for index,parameter in enumerate(initials):
    print(optimizer.circuitState())
    print(index,parameter)
    optimizer.circuit.initialization(parameter)  
    optimizer.parameters,optimizer.IDs = optimizer.circuitParameters()
    LBFGS_Delta.append(optimizer.minimization(lossFunction,flux_profile,
                    method='L-BFGS-B',subspace=(),options=dict(ftol=0,maxiter=15)))
    
Result = [optimizationAnalysis(initials,subspace,LBFGS_Delta)]
paths_delta,losse_delta = analysisPlotting(Result)

with open('ind_Insensitivity.pickle', 'wb') as handle:
    pickle.dump(LBFGS_Delta, handle, protocol=pickle.HIGHEST_PROTOCOL)

