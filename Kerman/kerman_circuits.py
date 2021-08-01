import numpy,utils
from circuit import Circuit
from components import J,C,L

def transmon(basis):
    transmon = [[0,1,{'J':J(0,1,100),'C':C(0,1,500),'L':L(0,1,100)}]]
    transmon = [J(0,1,100),C(0,1,500)]
    transmon = Circuit(transmon,basis)
    return transmon

def transmon(basis):
    transmon = [J(0,1,100),C(0,1,500),L(0,1,100,'I',True)]
    #transmon = [J(0,1,100),C(0,1,500)]
    transmon = Circuit(transmon,basis)
    return transmon

def transmonTunable(basis):
    return transmon

def shuntedQubit(basis):
    circuit = [[0,1,{'J':J(0,1,10),'C':C(0,1,10)}]]
    circuit += [[1,2,{'J':J(1,2,10),'C':C(1,2,10)}]]
    circuit += [[0,2,{'J':J(0,2,10),'C':C(0,2,10)}]]
    circuit = Circuit(circuit,basis)
    return circuit

def shuntedQubit(basis):
    circuit = [J(0,1,10),C(0,1,10)]
    circuit += [J(1,2,10),C(1,2,10)]
    circuit += [J(0,2,10),C(0,2,10)]
    circuit = Circuit(circuit,basis)
    return circuit

def phaseSlip():
    return circuit


if __name__=='__main__':
    circuit = transmon([7])
    utils.plotDOTGraph(circuit.G)
    #utils.plotDOTGraph(circuit.spanningTree)
    edges,Ej = circuit.josephsonComponents()
    spanning_tree = circuit.spanningTree()
    flux = circuit.loopFlux(0,1,'I56876',{'I':.3})
    H = circuit.josephsonEnergy({'I':.3})
