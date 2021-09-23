import numpy,utils,sys
from circuit import Circuit, hamiltonianEnergy, phase
from components import J,C,L,pi,h
from numpy.linalg import det
from pyvis import network as pvnet

def transmon(basis,tunable=True):
    transmon = [J(0,1,20),C(0,1,.3)]
    if tunable:
        transmon += [L(0,1,30,'I',True)]
    transmon = Circuit(transmon,basis)
    return transmon

def oscillatorLC(basis):
    oscillator = [L(0,1,.00031),C(0,1,51.6256)]
    return Circuit(oscillator,basis)

def shuntedQubit(basis):
    circuit = [J(1,2,10),C(1,2,10)]
    circuit += [J(2,3,10),C(2,3,5)]
    circuit += [J(3,0,10),C(3,0,2)]
    circuit += [L(0,1,10,'I',True)]

    circuit = Circuit(circuit,basis)
    return circuit

def phaseSlip(basis):
    circuit = [C(0,1,100)]
    circuit += [L(1,3,.001,'Ltl',True),L(1,4,.0005,'Lbl',True)]
    circuit += [J(3,2,10),J(4,2,10)]
    circuit += [C(3,2,30),C(4,2,30)]
    circuit += [C(2,5,30),C(2,6,30)]
    circuit += [J(2,5,10),J(2,6,100)]
    circuit += [C(2,0,40)]
    circuit += [L(5,7,.00002,'Ltr',True),L(6,7,.00035,'Lbr',True)]
    circuit += [L(1,7,.0005,'Ll',True)]
    circuit += [C(7,0,10)]
    circuit = Circuit(circuit,basis)
    return circuit

if __name__=='__main__':
    circuit = shuntedQubit([3,3,3])
    flux_manifold = zip(numpy.arange(0,1,.05))
    E0,spectrum = circuit.spectrumManifold(['I'],flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.05),{'excitation':spectrum,'ground_state':E0})
