from torch.optim import SGD,RMSprop,Adam,LBFGS,Rprop
import torch,pandas
from torch import tensor,argsort,zeros,abs,mean,stack,var,log,isnan,lobpcg,std
from torch.autograd import grad
from torch.linalg import det,inv,eigh as eigsolve, eigvalsh
from torch.nn.utils import clip_grad_norm_,clip_grad_value_
from numpy import arange,set_printoptions,meshgrid,linspace,array,sqrt,sort,log10,random,logspace
from scipy.optimize import minimize
from time import perf_counter,sleep
from DiSuQ.Torch.non_attacking_rooks import charPoly
from DiSuQ.Torch.components import L,J,C,L0,J0,C0
from DiSuQ.utils import empty
from scipy.stats import truncnorm

"""
    * Loss functions
    * Argument : dictionary{circuit Features}
    * Eigenvalues,Eigenvectors of Bottom 3 states
"""
MSE = torch.nn.MSELoss()

def anHarmonicity(spectrum):
    ground,Ist,IInd = spectrum[:3]
    return (IInd-Ist)-(Ist-ground)

def groundEnergy(spectrum):
    return spectrum[0]

class Optimization:
    def __init__(self,circuit,representation='K',sparse=False,algo=Adam):
        self.circuit = circuit
        self.parameters,self.IDs = self.circuitParameters()
        self.levels = [0,1,2]
        self.sparse = sparse
        self.representation = representation
        self.algo = algo
        
    def circuitID(self):
        IDs = []
        for component in self.circuit.network:
            IDs.append(component.ID)
        return IDs

    def circuitParameters(self,subspace=()):
        parameters = []; IDs = []
        slaves = self.circuit.pairs.keys()
        for component in self.circuit.network:
            if component.ID in subspace or len(subspace)==0:
                if component.__class__ == C :
                    parameter = component.cap
                elif component.__class__ == L :
                    parameter = component.ind
                elif component.__class__ == J :
                    parameter = component.jo
                if not component.ID in slaves:
                    IDs.append(component.ID)
                    parameters.append(parameter)
        return parameters,IDs
    
    def circuitState(self):
        circuit = self.circuit
        parameters = {}
        for component in circuit.network:
            parameters[component.ID] = component.energy().item()
        return parameters

    def parameterState(self):
        circuit = self.circuit
        parameters = {}
        for component in circuit.network:
            if component.__class__ == C :
                parameters[component.ID] = component.cap.item()
            elif component.__class__ == L :
                parameters[component.ID] = component.ind.item()
            elif component.__class__ == J :
                parameters[component.ID] = component.jo.item()
        return parameters

    def circuitHamiltonian(self,external_fluxes,to_dense=True):
        # returns Dense Hamiltonian
        if self.representation == 'K':
            H = self.circuit.kermanHamiltonianLC()
            H += self.circuit.kermanHamiltonianJosephson(external_fluxes)
        elif self.representation == 'Q':
            H = self.circuit.chargeHamiltonianLC()
            H += self.circuit.josephsonCharge(external_fluxes)
        elif self.representation == 'O':
            H = self.circuit.oscillatorHamiltonianLC()
            H += self.circuit.josephsonOscillator(external_fluxes)
        elif self.representation == 'R':
            H_J = self.circuit.ridgeJosephson(self.circuit)
            H = self.circuit.ridgeHamiltonianLC(self.circuit) + H_J(external_fluxes)
        if to_dense:
            H = H.to_dense()
        return H

class PolynomialOptimization(Optimization):
    """
        * multi-initialization starting : single point
        * sparse Hamiltonian exploration
        * Characteristic polynomial Root evaluation
        * variable parameter space : {C,L,J}
        * loss functions
        * circuit tuning
    """

    def __init__(self,circuit,representation='K',sparse=True):
        super().__init__(circuit,representation,sparse=True)

    def characterisiticPoly(self,H):
        # Non-Attacking Rooks algorithm
        # Implement multi-threading
        indices = H.coalesce().indices().T.detach().numpy()
        values = H.coalesce().values()
        N = len(H)
        data = dict(zip([tuple(index) for index in indices],values))
        coeffs = zeros(N+1); coeffs[0] = tensor(1.)
        stats = {'terminal':0,'leaf':0}
        poly = charPoly(coeffs,indices,N,data,stats)
        return poly

    def groundEigen(self,poly,start=tensor(-10.0),steps=5):
        # mid Netwon root minimum for characterisiticPoly
        # square polynomial
        # nested iteration
        return eigen

    def eigenState(self,energy,H):
        return state

    def spectrumOrdered(self,external_flux):
        H = self.circuitHamiltonian(external_flux)
        poly = self.characterisiticPoly(H)
        E0 = self.groundEigen(poly)
        state0 = self.groundState(E0,H)
        return E0,state0

    def optimization(self,loss_function,flux_profile,iterations=50,lr=.0001):
        # flux profile :: list of flux points dict{}
        # loss_function : list of Hamiltonian on all flux points
        logs = [];dParams = []
        optimizer = self.algo(self.parameters,lr=lr)
        start = perf_counter()
        for iter in range(iterations):
            optimizer.zero_grad()
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss = loss_function(Spectrum,flux_profile)
            logs.append({'loss':loss.detach().item(),'time':perf_counter()-start})
            dParams.append(self.parameterState())
            loss.backward()
            optimizer.step()

        dLog = pandas.DataFrame(logs)
        dLog['time'] = dLog['time'].diff()
        dLog.dropna(inplace=True)
        dParams = pandas.DataFrame(dParams)
        return logs

class OrderingOptimization(Optimization):
    """
        * multi-initialization starting : single point
        * Dense Hamiltonian exploration
        * Phase space correction
        * variable parameter space : {C,L,J}
        * loss functions
        * circuit tuning
    """
    def __init__(self,circuit,representation='K',sparse=False):
        super().__init__(circuit,representation,sparse)
        
    def stateOrdered(self,external_flux):
        H = self.circuitHamiltonian(external_flux)
        spectrum,state = eigsolve(H)
        spectrum = spectrum.real
        order = argsort(spectrum)#.clone().detach() # break point : retain graph
        return spectrum[order],state[order]

    def spectrumOrdered(self,external_flux):
        H = self.circuitHamiltonian(external_flux)
        if H.is_sparse:
            spectrum = lobpcg(H.to(float),k=4,largest=False)[0]; states = None
        else:
            #spectrum = eigvalsh(H); states = None
            spectrum,states = eigsolve(H)
        
        #spectrum = spectrum.real
        #order = argsort(spectrum)#.clone().detach() # break point : retain graph
        return spectrum,states

    def orderTransition(self,spectrum,order,levels=[0,1,2]):
        sorted = argsort(spectrum)
        if all(sorted[levels]==order[levels]):
            return order
        return sorted
    
    def lossScape(self,loss_function,flux_profile,scape,static):
        # parascape : {A:[...],B:[....]} | A,B in circuit.parameters
        A,B = scape.keys()
        Loss = empty((len(scape[A]),len(scape[B])))
        for id_A,a in enumerate(scape[A]):
            for id_B,b in enumerate(scape[B]):
                point = static.copy()
                point.update({A:a,B:b})
                self.circuit.initialization(point)
                Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
                loss,metrics = loss_function(Spectrum,flux_profile)
                Loss[id_A,id_B] = loss.detach().item()
        return Loss.transpose() # A -> X-axis , B -> Y-axis
        
    def minimization(self,loss_function,flux_profile,method='Nelder-Mead',subspace=(),options=dict()):
        x0 = self.circuitState()
        for slave,master in self.circuit.pairs.items():
            del x0[slave]
        static = dict() # static complement to subspace
        for key,value in x0.items():
            if not key in subspace and len(subspace)>0:
                static[key] = value
        for key in static:
            del x0[key]
        keys = tuple(x0.keys()) # subspace
        x0 =  array(tuple(x0.values())) # subspace
        def objective(parameters):
            parameters = dict(zip(keys,parameters))
            parameters.update(static)
            for slave,master in self.circuit.pairs.items():
                parameters[slave] = parameters[master]
            self.circuit.initialization(parameters)
            self.parameters,self.IDs = self.circuitParameters(subspace) 
            
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,metrics = loss_function(Spectrum,flux_profile)
            loss = loss.detach().item()
            return loss
        def gradient(parameters):
            parameters = dict(zip(keys,parameters))
            parameters.update(static)
            for slave,master in self.circuit.pairs.items():
                parameters[slave] = parameters[master]
            self.circuit.initialization(parameters)
            self.parameters,self.IDs = self.circuitParameters(subspace)
                 
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,metrics = loss_function(Spectrum,flux_profile)
            for parameter in self.parameters:
                if parameter.grad:
                    parameter.grad.zero_()
            #import pdb;pdb.set_trace()
            #if isnan(loss):
            #import pdb;pdb.set_trace()
            loss.backward(retain_graph=False)
            parameters = dict(zip(self.IDs,self.parameters))
            gradients = []
            for iD in keys:
                parameter = parameters[iD]
                if parameter.grad is None: 
                    # spectrum degeneracy:torch.eigh destablize
                    # loss detached from computation graph
                    #import ipdb;ipdb.set_trace()
                    gradients.append(0.0)
                else:
                    gradients.append(parameter.grad.detach().item())
            return gradients

        logs = []; dParams = [self.parameterState()]; dCircuit = [self.circuitState()]
        def logger(parameters):
            #parameters = dict(zip(keys,parameters))
            #parameters.update(static)
            #for slave,master in self.circuit.pairs.items():
            #    parameters[slave] = parameters[master]
            #self.circuit.initialization(parameters)
            #self.parameters,self.IDs = self.circuitParameters(subspace)
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,metrics = loss_function(Spectrum,flux_profile)
            loss = loss.detach().item()
            metrics['loss'] = loss
            metrics['time'] = perf_counter()-start
            logs.append(metrics)
            dParams.append(self.parameterState())
            dCircuit.append(self.circuitState())
        
        start = perf_counter()
        components = self.circuit.circuitComposition()
        Bounds = []
        for iD in keys:
            bound = components[iD].bounds()
            Bounds.append(bound) # positive boundary exclusive
        options['disp'] = True
        results = minimize(objective,x0,method=method,options=options,jac=gradient,bounds=Bounds,callback=logger)
        
        dLog = pandas.DataFrame(logs)
        if len(dLog)>0:
            dLog['time'] = dLog['time'].diff()
        else:
            #import pdb;pdb.set_trace()
            print('Failure initial')
        dParams = pandas.DataFrame(dParams)
        dCircuit = pandas.DataFrame(dCircuit)
        return dLog,dParams,dCircuit
    
    def lossModel(self,loss_function,flux_profile):
        def loss(parameters):
            # parameters !!!!
            # circuit object restructuring 
            circuit.initialization(parameters)
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,_ = loss_function(Spectrum,flux_profile)
            return loss
        return loss
    
    def optimizationLBFGS(self,loss_function,flux_profile,max_iter=20,history_size=100,tol=1e-6,iterations=100,lr=None,log=False):
        # flux profile :: list of flux points dict{}
        # loss_function : list of Hamiltonian on all flux points
        logs = []; dParams = []; dCircuit = []
        if lr is None:
            optimizer = LBFGS(self.parameters,max_iter=max_iter,history_size=history_size,tolerance_change=tol,
                              line_search_fn='strong_wolfe')
        else :
            optimizer = LBFGS(self.parameters,lr=lr,max_iter=max_iter,history_size=history_size,tolerance_change=tol,
                              line_search_fn=None)
        def closure():
            optimizer.zero_grad()
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,_ = loss_function(Spectrum,flux_profile)                
            loss.backward(retain_graph=True)
            return loss
        
        dParams.append(self.parameterState())
        dCircuit.append(self.circuitState())
        start = perf_counter()
        for epoch in range(iterations):            
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,_ = loss_function(Spectrum,flux_profile)
            metrics = dict()
            metrics['loss'] = loss.detach().item()
            optimizer.step(closure)
            metrics['time'] = perf_counter()-start
            dParams.append(self.parameterState())
            dCircuit.append(self.circuitState())
            if log:
                spectrum = dict()
                for level in range(3):
                    spectrum['0-level-'+str(level)] = Spectrum[0][0][level].detach().item()
                    spectrum['pi-level-'+str(level)] = Spectrum[int(len(flux_profile)/2)][0][level].detach().item()
                metrics.update(spectrum)
                gradients = [parameter.grad.detach().item() for parameter in self.parameters]
                gradients = dict(zip(['grad-'+ID for ID in self.IDs],gradients))
                metrics.update(gradients)
                    #hess = hessian(self.lossModel(loss_function,flux_profile))
                    #hess = tensor(hess)                    
                    #hess = dict(zip(['hess'+index for index in range(len(hess))],hess))
                    #metrics.update(hess)
                #print([parameter.grad for parameter in self.parameters])
            
            logs.append(metrics)
            if breakPoint(logs[-15:]):
                print('Optimization Break Point xxxxxx')
                break

        dLog = pandas.DataFrame(logs)
        dLog['time'] = dLog['time'].diff()
        dParams = pandas.DataFrame(dParams)
        dCircuit = pandas.DataFrame(dCircuit)
        return dLog,dParams,dCircuit

    def optimization(self,loss_function,flux_profile,iterations=100,lr=1e-5):
        # flux profile :: list of flux points dict{}
        # loss_function : list of Hamiltonian on all flux points
        logs = []; dParams = []; dCircuit = []
        optimizer = self.algo(self.parameters,lr=lr)
        dParams.append(self.parameterState())
        dCircuit.append(self.circuitState())
        start = perf_counter()
        for epoch in range(iterations):            
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,metrics = loss_function(Spectrum,flux_profile)
            metrics['loss'] = loss.detach().item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            metrics['time'] = perf_counter()-start
            dParams.append(self.parameterState())
            dCircuit.append(self.circuitState())
            logs.append(metrics)
            if breakPoint(logs[-15:]):
                print('Optimization Break Point xxxxxx')
                break

        dLog = pandas.DataFrame(logs)
        dLog['time'] = dLog['time'].diff()
        dParams = pandas.DataFrame(dParams)
        dCircuit = pandas.DataFrame(dCircuit)
        return dLog,dParams,dCircuit
    
def breakPoint(logs):
    # break optimization loop : stagnation / spiking
    loss = pandas.DataFrame(logs)['loss'].to_numpy()
    if loss[-1] > 1e12 and len(loss) > 10:
        return True    
    return False

def truncNormalParameters(circuit,subspace,N,var=5):
    # var : std of normal distribution
    iDs,domain = [],[]
    for component in circuit.network:
        if component.ID in subspace:
            iDs.append(component.ID)
            loc = component.energy().item()
            a,b = component.bounds()
            a = (a - loc)/var ; b = (b - loc)/var
            domain.append(truncnorm.rvs(a,b,loc,var,size=N,random_state=random.seed(101)))
    grid = array(domain)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(iDs,point)))
        space.append(state)
    return space

def uniformParameters(circuit,subspace,N):
    iDs,domain = [],[]
    for component in circuit.network:
        if component.ID in subspace:
            iDs.append(component.ID)
            a,b = component.bounds()
            a = log10(a.item()); b = log10(b.item())
            domain.append(logspace(a,b,N+1,endpoint=False)[1:])
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(iDs),-1)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(iDs,point)))
        space.append(state)
    return space

def domainParameters(domain,circuit,subspace):
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(subspace),-1)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(subspace,point)))
        space.append(state)
    return space

def initializationSequential(parameters,optimizer,lossFunction,flux_profile,iterations=100,lr=.005):
    Search = []
    for index,parameter in enumerate(parameters):
        optimizer.circuit.initialization(parameter)
        optimizer.parameters,_ = optimizer.circuitParameters()
        Search.append(optimizer.optimization(lossFunction,flux_profile,iterations=iterations,lr=lr))
    return Search
    
def initializationParallelism(optimizer,lossFunction,flux_profile,iterations=100,lr=.005):
    def optimization(parameters):
        optimizer.circuit.initialization(parameters)
        optim.parameters,_ = optim.circuitParameters()
        return optimizer.optimization(lossFunction,flux_profile,iterations=iterations,lr=lr)
    return optimization
    
def anHarmonicityProfile(optim,flux_profile):
    anHarmonicity_profile = []
    for flux in flux_profile:
        spectrum,state = optim.spectrumOrdered(flux)
        anHarmonicity_profile.append(anHarmonicity(spectrum).item())
    return anHarmonicity_profile

# Full batch of flux_profile
def lossTransitionFlatness(Spectrum,flux_profile):
    spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
    loss = var(spectrum[:,1]-spectrum[:,0])
    loss += var(spectrum[:,2]-spectrum[:,1])
    return loss,dict()
        
# def std(E):
#     N = len(E)
#     profile = tensor([0.]*N)
#     for index,e in enumerate(E):
#         profile += e
    

def lossDegeneracyWeighted(delta0,D0,N=2):
    def Loss(Spectrum,flux_profile):
        spot = 0
        
        E10 = [E[0][1]-E[0][0] for E in Spectrum]
        E20 = [E[0][2]-E[0][0] for E in Spectrum]
        
        Ist = abs(E10[0]-E10[1])
        #E10 = tensor(E10,requires_grad=True)

        # n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        #        degen_point = flux_profile[0]['Lx']
        #delta = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]-e10
        #delta = delta/e10
        #delta = delta.abs()
        #sensitivity = grad(e10,degen_point,create_graph=True)[0] # local fluctuations
        #sensitivity = log(sensitivity.abs())
        
        D = log((E20[0])/(E10[0]))/log(tensor(10.))
        delta = log(Ist/E10[0])/log(tensor(10.))
        loss = delta*delta0 - D*D0
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':E10[0].detach().item(),'E20':E20[0].detach().item()}
    return Loss

def lossDegeneracyTarget(delta0,D0):
    def Loss(Spectrum,flux_profile):
        half = 0#int(len(flux_profile)/2)
        neighbour = -1
        e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
        e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
        D = log(e20/e10)
        degen_point = flux_profile[0]['Lx']
        n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        delta = log((n10-e10).abs()/e20)
        loss = (delta0+delta)**2 + (D0-D)**2
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':e10.detach().item(),'E20':e20.detach().item()}
    return Loss

# def lossDegeneracy(Spectrum,flux_profile):
#     half = 0#int(len(flux_profile)/2)
#     neighbour = -1
#     e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
#     e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
#     D = log(e20/e10)
#     degen_point = flux_profile[0]['Lx']
#     n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
#     delta = log((n10-e10).abs())
#     #import pdb;pdb.set_trace()
#     a = 1/(log(n10).abs()).detach().item() ; b = 1/(log(e10).abs()).detach().item()
#     k = sqrt(2/(a**2+b**2))
#     loss = k*(a*log(n10) + b*log(e10))
#     return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':e10.detach().item()}


# def lossDegeneracy(Spectrum,flux_profile):
#     half = 0#int(len(flux_profile)/2)
#     neighbour = -1
#     e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
#     e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
#     D = log(e20/e10)
#     degen_point = flux_profile[0]['Lx']
#     n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
#     loss = log(n10)
#     return loss,{'D':D.detach().item(),'N10':n10.detach().item(),'E10':e10.detach().item()}

def lossAnharmonicity(alpha):
    def lossFunction(Spectrum,flux_profile):
        anharmonicity = tensor(0.0)
        for spectrum,state in Spectrum:
            anharmonicity += anHarmonicity(spectrum)
        anharmonicity = anharmonicity/len(Spectrum)
        loss = MSE(anharmonicity,tensor(alpha))
        return loss,{'anharmonicity':anharmonicity.detach().item()}
    return lossFunction

def lossE10(E10):
    def lossFunction(Spectrum,flux_profile):
        loss = tensor(0.0)
        spectrum = stack([spectrum[:2] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        loss += MSE(e10,tensor(E10))
        return loss,dict()
    return lossFunction

def lossTransition(E10,E20):
    def lossFunction(Spectrum,flux_profile):
        spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        e20 = spectrum[:,2]-spectrum[:,0]
        loss = MSE(e10,E10) + MSE(e20,E20)
        log = {'mid10':e10[int(len(flux_profile)/2)].detach().item(),'mid20':e20[int(len(flux_profile)/2)].detach().item()}
        return loss,log
    return lossFunction

# Stochastic sample on flux_profile
#def loss_Anharmonicity(Spectrum,flux):
#    spectrum,state = Spectrum
#    return MSE(anHarmonicity(spectrum),tensor(.5))

if __name__=='__main__':
    basis = [15]
    from models import transmon
    circuit = transmon(basis,sparse=False)
    optim = OrderingOptimization(circuit,representation='Q')
    print(circuit.circuitComponents())
    flux_profile = [dict()]
    
    dLogs,dParams,dCircuit = optim.optimization(loss_Energy,flux_profile,lr=.00001)
    print("refer Optimization Verification")
