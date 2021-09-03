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

def phaseSlip():
    return circuit

if __name__=='__main__':
    circuit = oscillatorLC([2])
    eigenenergies = circuit.circuitEnergy(dict())
    print(eigenenergies[1])
    sys.exit(0)
    circuit = transmon([2],False)
    eigenenergies = circuit.circuitEnergy(dict())
    print(eigenenergies[0])
    sys.exit(0)
    circuit = transmon([2])
    utils.plotMatPlotGraph(circuit.G,'circuit')
    utils.plotMatPlotGraph(circuit.spanning_tree,'spanning_tree')
    flux_manifold = zip(numpy.arange(0,1,.01))
    E0,spectrum = circuit.spectrumManifold(['I'],flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.01),{'excitation':spectrum,'ground_state':E0})
