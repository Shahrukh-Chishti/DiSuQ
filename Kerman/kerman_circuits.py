import numpy
from circuit import Circuit
from components import J,C,L

def transmon(basis):
    transmon = [[0,1,{'J':J(0,1,100),'C':C(0,1,500),'L':L(0,1,100)}]]
    transmon = Circuit(transmon,basis)
    return transmon

def shuntedQubit(basis):
    circuit = [[0,1,{'J':J(0,1,10),'C':C(0,1,10)}]]
    circuit += [[1,2,{'J':J(1,2,10),'C':C(1,2,10)}]]
    circuit += [[0,2,{'J':J(0,2,10),'C':C(0,2,10)}]]
    circuit = Circuit(circuit,basis)
    return circuit

def phaseSlip():
    return circuit

def qucat2Kerman(circuit):
    return circuit

if __name__=='__main__':
    circuit = transmon([7])
    H = circuit.hamiltonian_charged([1])
