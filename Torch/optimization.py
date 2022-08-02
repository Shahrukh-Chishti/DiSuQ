from torch.optim import SGD,RMSprop,Adam
import models,torch,pandas
from torch import tensor,argsort
from torch.linalg import det,inv,eig as eigsolve
from numpy import arange,set_printoptions
from time import perf_counter,sleep
import components

"""
    * Loss functions
    * Argument : dictionary{circuit Features}
    * Eigenvalues,Eigenvectors of Bottom 3 states
"""
MSE = torch.nn.MSELoss()

def anHarmonicity(spectrum):
    ground,Ist,IInd = spectrum[:3]
    return (IInd-Ist)/(Ist-ground)

def groundEnergy(spectrum):
    return spectrum[0]

class PolynomialOptimization:
    """
        * multi-initialization starting : single point
        * sparse Hamiltonian exploration
        * Characteristic polynomial Root evaluation
        * variable parameter space : {C,L,J}
        * loss functions
        * circuit tuning
    """

    def __init__(self,circuit):
        self.circuit = circuit
        self.parameters = self.circuitParameters()

    def circuitParameters(self):
        circuit = self.circuit
        parameters = []
        for component in circuit.network:
            if component.__class__ == components.C :
                parameters.append(component.capacitance)
            elif component.__class__ == components.L :
                parameters.append(component.inductance)
            elif component.__class__ == components.J :
                parameters.append(component.energy)
        return parameters

    def characterisiticPoly(self,H):
        # Non-Attacking Rooks algorithm
        # Implement multi-threading
        indices = H.coalesce().indices().T.detach().numpy()
        values = H.coalesce().values()

        return poly

    def groundEigen(self,poly,start=tensor(-10.0),steps=5):
        # mid Netwon root minimum for characterisiticPoly
        # square polynomial
        # nested iteration
        return eigen

    def groundState(self,ground_eigen):
        return state

    def optimization(self,loss_function,flux_profile,iterations=50,lr=.0001):
        # flux profile :: list of flux points dict{}
        # loss_function : list of Hamiltonian on all flux points
        logs = []
        optimizer = SGD(self.parameters,lr=lr)
        optimizer.zero_grad()
        for iter in range(iterations):
            Spectrum = [self.groundEigen(flux) for flux in flux_profile]
            loss = loss_function(Spectrum,flux_profile)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs.append({'loss':loss.detach().numpy()})
        return logs

class OrderingOptimization:
    """
        * multi-initialization starting : single point
        * Dense Hamiltonian exploration
        * Phase space correction
        * variable parameter space : {C,L,J}
        * loss functions
        * circuit tuning
    """
    def __init__(self,circuit,sparse=False):
        self.circuit = circuit
        self.parameters = self.circuitParameters()
        self.levels = [0,1,2]
        self.sparse = sparse

    def circuitParameters(self):
        circuit = self.circuit
        parameters = []
        for component in circuit.network:
            if component.__class__ == components.C :
                parameters.append(component.cap)
            elif component.__class__ == components.L :
                parameters.append(component.ind)
            elif component.__class__ == components.J :
                parameters.append(component.jo)
        return parameters

    def parameterState(self):
        circuit = self.circuit
        parameters = {}
        for component in circuit.network:
            if component.__class__ == components.C :
                parameters[component.ID] = component.cap.item()
            elif component.__class__ == components.L :
                parameters[component.ID] = component.ind.item()
            elif component.__class__ == components.J :
                parameters[component.ID] = component.jo.item()
        return parameters

    def circuitHamiltonian(self,external_fluxes):
        # returns Dense Kerman hamiltonian
        H = self.circuit.kermanHamiltonianLC()
        H += self.circuit.kermanHamiltonianJosephson(external_fluxes)
        if self.sparse:
            H = H.to_dense()
        return H

    def spectrumOrdered(self,external_fluxes):
        H = self.circuitHamiltonian(external_fluxes)
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
        logs = []
        dParams = []
        optimizer = SGD(self.parameters,lr=lr)
        start = perf_counter()
        for epoch in range(iterations):
            optimizer.zero_grad()
            Spectrum = [self.spectrumOrdered(flux) for flux in flux_profile]
            #Spectrum = self.spectrumOrdered(flux)
            loss = loss_function(Spectrum,flux_profile)
            logs.append({'loss':loss.detach().item(),'time':perf_counter()-start})
            dParams.append(self.parameterState())
            loss.backward(retain_graph=True)
            optimizer.step()

        dLog = pandas.DataFrame(logs)
        dLog['time'] = dLog['time'].diff()
        dLog.dropna(inplace=True)
        dParams = pandas.DataFrame(dParams)
        return dLog,dParams

def anHarmonicityProfile(optim,flux_profile):
    anHarmonicity_profile = []
    for flux in flux_profile:
        spectrum,state = optim.spectrumOrdered(flux)
        anHarmonicity_profile.append(anHarmonicity(spectrum).item())
    return anHarmonicity_profile

# Full batch of flux_profile
def loss_Anharmonicity(Spectrum,flux_profile):
    loss = tensor(0.0)
    for spectrum,state in Spectrum:
        loss += MSE(anHarmonicity(spectrum),tensor(10.))
    loss = loss/len(Spectrum)
    return loss

# Stochastic sample on flux_profile
#def loss_Anharmonicity(Spectrum,flux):
#    spectrum,state = Spectrum
#    return MSE(anHarmonicity(spectrum),tensor(.5))

if __name__=='__main__':
    basis = {'O':[2],'I':[],'J':[2,2]}
    circuit = models.shuntedQubit(basis)
    optim = OrderingOptimization(circuit,sparse=False)
    flux_profile = tensor(arange(0,1,.1))
    flux_profile = [{'I':flux} for flux in flux_profile]
    print(circuit.circuitComponents())
    print(anHarmonicityProfile(optim,flux_profile))
    dLog = optim.optimization(loss_Anharmonicity,flux_profile,iterations=100,lr=.0001)
    print(circuit.circuitComponents())
    print(anHarmonicityProfile(optim,flux_profile))
    print(dLog)
