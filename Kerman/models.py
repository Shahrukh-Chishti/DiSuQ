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

def shuntedQubit(basis,josephson=[10.,15.,20.],cap=[100.,500.,200.],ind=.0001):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(1,2,Ej1),C(1,2,C1)]
    circuit += [J(2,3,Ej2),C(2,3,C2)]
    circuit += [J(3,0,Ej3),C(3,0,C3)]
    circuit += [L(0,1,ind,'I',True)]

    circuit = Circuit(circuit,basis)
    return circuit

def phaseSlip(basis,inductance=[.001,.0005,.00002,.00035,.0005],capacitance=[100,30,30,30,30,40,10]):
    La,Lb,Lc,Ld,Le = inductance
    Ca,Cb,Cc,Cd,Ce,Cf,Cg = capacitance
    circuit = [C(0,1,Ca)]
    circuit += [L(1,3,La,'Ltl',True),L(1,4,Lb,'Lbl',True)]
    circuit += [J(3,2,10),J(4,2,10)]
    circuit += [C(3,2,Cb),C(4,2,Cc)]
    circuit += [C(2,5,Cd),C(2,6,Ce)]
    circuit += [J(2,5,10),J(2,6,100)]
    circuit += [C(2,0,Cf)]
    circuit += [L(5,7,Lc,'Ltr',True),L(6,7,Ld,'Lbr',True)]
    circuit += [L(1,7,Le,'Ll',True)]
    circuit += [C(7,0,Cg)]
    circuit = Circuit(circuit,basis)
    return circuit

if __name__=='__main__':
    circuit = shuntedQubit([5,5,5])
    flux_manifold = zip(numpy.arange(0,1,.05))
    import ipdb; ipdb.set_trace()
    E0,Ex = circuit.spectrumManifold(['I'],flux_manifold,H_LC=circuit.kermanHamiltonianLC())
    utils.plotCompare(numpy.arange(0,1,.05),{'excitation':Ex,'ground_state':E0})
