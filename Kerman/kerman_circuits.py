import numpy,utils,sys
from circuit import Circuit, hamiltonianEnergy, phase
from components import J,C,L
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

if __name__=='__main__':
    circuit = phaseSlip([1,1,1,1,1,1,1])
    bias_flux = {'Ltl':.3,'Lbl':.1,'Ltr':.09,'Lbr':.23,'Ll':.45}
    utils.plotMatPlotGraph(circuit.G,'circuit')
    utils.plotMatPlotGraph(circuit.spanning_tree,'spanning_tree')
    eigenenergies = circuit.circuitEnergy(bias_flux)
    print(eigenenergies)
    sys.exit(0)
    circuit = oscillatorLC([2])
    eigenenergies = circuit.circuitEnergy(dict())
    print(eigenenergies[1])
    sys.exit(0)
    circuit = transmon([2],False)
    eigenenergies = circuit.circuitEnergy(dict())
    print(eigenenergies[0])
    sys.exit(0)
    circuit = transmon([2])
    flux_manifold = zip(numpy.arange(0,1,.01))
    E0,spectrum = circuit.spectrumManifold(['I'],flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.01),{'excitation':spectrum,'ground_state':E0})
