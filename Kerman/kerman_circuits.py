import numpy,utils,sys
from circuit import Circuit, hamiltonianEnergy, phase
from components import J,C,L,pi,h
from numpy.linalg import det
from pyvis import network as pvnet

def transmon(basis,tunable=True):
    transmon = [J(0,1,1000),C(0,1,50000)]
    if tunable:
        transmon += [L(0,1,10000,'I',True)]
    transmon = Circuit(transmon,basis)
    return transmon

def oscillatorLC(basis):
    oscillator = [L(0,1,1000),C(0,1,4000)]
    return Circuit(oscillator,basis)

def shuntedQubit(basis):
    circuit = [J(1,2,10),C(1,2,1000)]
    circuit += [J(2,3,100),C(2,3,5500)]
    circuit += [J(3,0,150),C(3,0,2500)]
    circuit += [L(0,1,10,'I',True)]

    circuit = Circuit(circuit,basis)
    return circuit

def phaseSlip(basis):
    circuit = [C(0,1,10000)]
    circuit += [L(1,3,10,'Ltl',True),L(1,4,50,'Lbl',True)]
    circuit += [J(3,2,100),J(4,2,100)]
    circuit += [C(3,2,3000),C(4,2,3000)]
    circuit += [C(2,5,3000),C(2,6,3000)]
    circuit += [J(2,5,100),J(2,6,100)]
    circuit += [C(2,0,4000)]
    circuit += [L(5,7,20,'Ltr',True),L(6,7,35,'Lbr',True)]
    circuit += [L(1,7,50,'Ll',True)]
    circuit += [C(7,0,10000)]
    circuit = Circuit(circuit,basis)
    return circuit

def testPhaseSlip():
    circuit = phaseSlip([1,1,1,1,1,1,1])
    flux_range = numpy.arange(0,1,.01)
    flux_static = [.01]*len(flux_range)
    flux_static = [flux_static]*4
    flux_manifold = zip(*flux_static,flux_range) # variation in Ll
    E0,spectrum = circuit.spectrumManifold(['Ltl','Lbl','Ltr','Lbr','Ll'],flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.01),{'excitation':spectrum,'ground_state':E0})

if __name__=='__main__':
    circuit = shuntedQubit([4,4,4])
    flux_manifold = zip(numpy.arange(0,1,.05))
    E0,spectrum = circuit.spectrumManifold(['I'],flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.05),{'excitation':spectrum,'ground_state':E0})
