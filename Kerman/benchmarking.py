import networkx,qucat,numpy
import matplotlib.pyplot as plt
#from qucat import Network,L,J,C,R
from circuit import Circuit
from components import *
import plotly.offline as py
import plotly.graph_objs as go

def transmon():
    transmon = [[0,1,{'J':J(0,1,10),'C':C(0,1,5),'L':L(0,1,100)}]]
    transmon = Circuit(transmon)
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
    circuit = shuntedQubit([5,5])
    spectrum = []
    fluxes = numpy.linspace(-1,1,100)
    for flux in fluxes:
        H = circuit.hamiltonian_charged([0,0,flux])
        energy = numpy.linalg.eigvals(H)
        spectrum.append(min(energy).real)
    py.plot([go.Scatter(x=fluxes,y=spectrum)])
    #networkx.draw_spring(circuit.G)
    #plt.show()

