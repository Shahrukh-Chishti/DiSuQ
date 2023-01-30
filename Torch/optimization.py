from torch.optim import SGD,RMSprop,Adam
import torch,pandas
from torch import tensor,argsort,zeros,abs,mean,stack,var
from torch.linalg import det,inv,eig as eigsolve
from numpy import arange,set_printoptions,meshgrid,linspace,array
from time import perf_counter,sleep
from DiSuQ.Torch.non_attacking_rooks import charPoly
from DiSuQ.Torch.components import L,J,C
from DiSuQ.Torch.components import L0,J0,C0


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
    def __init__(self,circuit,representation='K',sparse=False):
        self.circuit = circuit
        self.parameters = self.circuitParameters()
        self.levels = [0,1,2]
        self.sparse = sparse
        self.representation = representation

    def circuitParameters(self):
        circuit = self.circuit
        parameters = []
        for component in circuit.network:
            if component.__class__ == C :
                parameters.append(component.cap)
            elif component.__class__ == L :
                parameters.append(component.ind)
            elif component.__class__ == J :
                parameters.append(component.jo)
        return parameters
    
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

    def circuitHamiltonian(self,external_fluxes,to_dense=False):
        # returns Dense Hamiltonian
        if self.representation == 'K':
            H = self.circuit.kermanHamiltonianLC()
            H += self.circuit.kermanHamiltonianJosephson(external_fluxes)
        elif self.representation == 'Q':
            H = self.circuit.chargeHamiltonianLC()
            H += self.circuit.josephsonCharge(external_fluxes)
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
        optimizer = Adam(self.parameters,lr=lr)
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

    def spectrumOrdered(self,external_flux):
        H = self.circuitHamiltonian(external_flux)
        spectrum,state = eigsolve(H)
        spectrum = spectrum.real
        order = argsort(spectrum)#.clone().detach() # break point : retain graph
        return spectrum[order],state[order]

    def phaseTransition(self,spectrum,order,levels=[0,1,2]):
        sorted = argsort(spectrum)
        if all(sorted[levels]==order[levels]):
            return order
        return sorted

    def optimization(self,loss_function,flux_profile,iterations=100,lr=1e-5):
        # flux profile :: list of flux points dict{}
        # loss_function : list of Hamiltonian on all flux points
        logs = []; dParams = []; dCircuit = []
        optimizer = SGD(self.parameters,lr=lr)
        start = perf_counter()
        for epoch in range(iterations):
            optimizer.zero_grad()
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            loss,metrics = loss_function(Spectrum,flux_profile)          
            loss.backward(retain_graph=True)
            optimizer.step()
            metrics['loss'] = loss.detach().item()
            metrics['time'] = perf_counter()-start
            dParams.append(self.parameterState())
            dCircuit.append(self.circuitState())
            logs.append(metrics)

        dLog = pandas.DataFrame(logs)
        dLog['time'] = dLog['time'].diff()
        dLog.dropna(inplace=True)
        dParams = pandas.DataFrame(dParams)
        dCircuit = pandas.DataFrame(dCircuit)
        return dLog,dParams,dCircuit
    
def uniformParameters(circuit,N):
    iDs,domain = [],[]
    for component in circuit.network:
        iDs.append(component.ID)
        if component.__class__ == C :
            bound = C0
        elif component.__class__ == L :
            bound = L0
        elif component.__class__ == J :
            bound =J0
        domain.append(linspace(0,bound,N,endpoint=False)[1:])
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(iDs),-1)
    return dict(zip(iDs,grid))      
    
def initializationParallelism(optimizer,lossFunction,flux_profile,iterations=100,lr=.005):
    def optimization(parameters):
        optimizer.circuit.initialization(parameters)
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

def loss_Anharmonicity(alpha):
    def lossFunction(Spectrum,flux_profile):
        anharmonicity = tensor(0.0)
        for spectrum,state in Spectrum:
            anharmonicity += anHarmonicity(spectrum)
        anharmonicity = anharmonicity/len(Spectrum)
        loss = MSE(anharmonicity,tensor(alpha))
        return loss,{'anharmonicity':anharmonicity.detach().item()}
    return lossFunction

def loss_E10(E10):
    def lossFunction(Spectrum,flux_profile):
        loss = tensor(0.0)
        spectrum = stack([spectrum[:2] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        loss += MSE(e10,tensor(E10))
        return loss,dict()
    return lossFunction

def loss_Transition(E10,E21):
    def lossFunction(Spectrum,flux_profile):
        spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        e21 = spectrum[:,2]-spectrum[:,1]
        loss = MSE(e10,E10) + MSE(e21,E21)
        log = {'mid10':e10[int(len(flux_profile)/2)].detach().item(),'mid21':e21[int(len(flux_profile)/2)].detach().item()}
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
