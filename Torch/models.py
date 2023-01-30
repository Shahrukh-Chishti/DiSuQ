import numpy,sys
from DiSuQ.Torch.circuit import Circuit, hamiltonianEnergy, phase
from DiSuQ.Torch.components import J,C,L,pi,h
from DiSuQ.Torch.components import C0,J0,L0
from numpy.linalg import det
from pyvis import network as pvnet
from torch import tensor

def tensorize(values,variable=True):
    tensors = []
    for val in values:
        tensors.append(tensor(val,requires_grad=variable))
    return tensors

def sigInv(sig,limit):
    return [sigmoidInverse(s/limit) for s in sig]

def transmon(basis,Ej=10.,Ec=0.3,sparse=True):
    transmon = [J(0,1,Ej,'J')]
    transmon += [C(0,1,Ec,'C')]

    transmon = Circuit(transmon,basis,sparse)
    return transmon

def splitTransmon(basis):
    transmon = [J(0,1,10),C(0,1,100)]
    transmon += [L(1,2,.0003,'I',True)]
    transmon += [J(2,0,10),C(2,0,100)]
    transmon = Circuit(transmon,basis)
    return transmon

def oscillatorLC(basis,El=.00031,Ec=51.6256,sparse=True):
    oscillator = [L(0,1,El),C(0,1,Ec)]
    return Circuit(oscillator,basis,sparse)

def fluxoniumArray(basis,junc=.05,N=2,Ec=100,Ej=150,sparse=True):
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'Junc')]
    for i in range(N):
        circuit += [J(1+i,2+i,junc,'junc'+str(i))]
    circuit += [J(1+N,0,junc,'junc'+str(N))]
    return circuit

def fluxonium(basis,El=.0003,Ec=100,Ej=20,sparse=True):
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'JJ')]
    circuit += [L(0,1,El,'I',True)]

    circuit = Circuit(circuit,basis,sparse)
    return circuit

def shuntedQubit(basis,josephson=[450.,15.,20.],cap=[.01,.01,.02],ind=.00001,sparse=True):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(1,2,Ej1,'JJ1'),C(1,2,C1,'C1')]
    circuit += [J(2,3,Ej2,'JJ2'),C(2,3,C2,'C2')]
    circuit += [J(3,0,Ej3,'JJ3'),C(3,0,C3,'C3')]
    circuit += [L(0,1,ind,'I',True)]

    circuit = Circuit(circuit,basis,sparse)
    return circuit

def shuntedQubitFluxFree(basis,josephson=[10.,15.,20.],cap=[100.,500.,200.],sparse=True):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(0,1,Ej1,'JJ1'),C(0,1,C1,'C1')]
    circuit += [J(1,2,Ej2,'JJ2'),C(1,2,C2,'C2')]
    circuit += [J(2,0,Ej3,'JJ3'),C(2,0,C3,'C3')]

    circuit = Circuit(circuit,basis,sparse)
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
    circuit = shuntedQubit()
    print(circuit.modeDistribution())
    circuit.basis = {'O':[10],'I':[0],'J':[10,10]}
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson({'I':tensor(.25)})

    import ipdb; ipdb.set_trace()
