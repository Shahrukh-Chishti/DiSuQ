import numpy,utils,sys
from circuit import Circuit, hamiltonianEnergy, phase
from components import J,C,L,pi,h
from numpy.linalg import det
from pyvis import network as pvnet

def transmon(basis,Ej=10,Ec=0.3):
    transmon = [J(0,1,Ej)]
    transmon += [C(0,1,Ec)]

    transmon = Circuit(transmon,basis)
    return transmon

def splitTransmon(basis):
    transmon = [J(0,1,10),C(0,1,100)]
    transmon += [L(1,2,.0003,'I',True)]
    transmon += [J(2,0,10),C(2,0,100)]
    transmon = Circuit(transmon,basis)
    return transmon

def oscillatorLC(basis,El=.00031,Ec=51.6256):
    oscillator = [L(0,1,El),C(0,1,Ec)]
    return Circuit(oscillator,basis)

def shuntedQubit(basis):
    circuit = [J(1,2,10),C(1,2,100)]
    circuit += [J(2,3,10),C(2,3,500)]
    circuit += [J(3,0,10),C(3,0,200)]
    circuit += [L(0,1,.0001,'I',True)]

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
    circuit = shuntedQubit([5,5,5])
    flux_manifold = zip(numpy.arange(0,1,.05))
    import ipdb; ipdb.set_trace()
    E0,Ex = circuit.spectrumManifold(['I'],flux_manifold,H_LC=circuit.kermanHamiltonianLC())
    utils.plotCompare(numpy.arange(0,1,.05),{'excitation':Ex,'ground_state':E0})
