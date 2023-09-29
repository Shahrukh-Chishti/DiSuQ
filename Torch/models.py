import numpy,sys
from DiSuQ.Torch.circuit import Circuit, hamiltonianEnergy, phase
from DiSuQ.Torch.components import J,C,L,pi,h
from DiSuQ.Torch.components import C0,J0,L0,capE,indE,C_,J_,L_
from DiSuQ.Torch.components import e,h,flux_quanta,hbar
from numpy.linalg import det
from pyvis import network as pvnet
from torch import tensor,diag,sqrt
from torch.linalg import inv

def tensorize(values,variable=True):
    tensors = []
    for val in values:
        tensors.append(tensor(val,requires_grad=variable))
    return tensors

def sigInv(sig,limit):
    return [sigmoidInverse(s/limit) for s in sig]

# input order enumerates into node index

def zeroPi(basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,sparse=True,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0)):
    circuit = [L(1,4,El,'Lx',True,_L_[1],_L_[0]),L(2,3,El,'Ly',True,_L_[1],_L_[0])]
    circuit += [C(1,3,Ec,'Cx',_C_[1],_C_[0]),C(2,4,Ec,'Cy',_C_[1],_C_[0])]
    circuit += [J(3,4,Ej,'Jx',_J_[1],_J_[0]),J(2,1,Ej,'Jy',_J_[1],_J_[0])]
    circuit += [C(3,4,EcJ,'CJx',_CJ_[1],_CJ_[0]),C(2,1,EcJ,'CJy',_CJ_[1],_CJ_[0])]
    pairs = dict()
    if symmetry:
        pairs['Ly'] = 'Lx'
        pairs['Cy'] = 'Cx'
        pairs['Jy'] = 'Jx'
        pairs['CJy'] = 'CJx'
    
    circuit = Circuit(circuit,basis,sparse,pairs)
    return circuit

def zeroPi(basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,sparse=True,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0)):
    circuit = [L(0,1,El,'Lx',True,_L_[1],_L_[0]),L(2,3,El,'Ly',True,_L_[1],_L_[0])]
    circuit += [C(1,2,Ec,'Cx',_C_[1],_C_[0]),C(3,0,Ec,'Cy',_C_[1],_C_[0])]
    circuit += [J(1,3,Ej,'Jx',_J_[1],_J_[0]),J(2,0,Ej,'Jy',_J_[1],_J_[0])]
    circuit += [C(1,3,EcJ,'CJx',_CJ_[1],_CJ_[0]),C(2,0,EcJ,'CJy',_CJ_[1],_CJ_[0])]
    pairs = dict()
    if symmetry:
        pairs['Ly'] = 'Lx'
        pairs['Cy'] = 'Cx'
        pairs['Jy'] = 'Jx'
        pairs['CJy'] = 'CJx'
    
    circuit = Circuit(circuit,basis,sparse,pairs)
    return circuit

def zeroPi(basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,sparse=True,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0),ridge = False,flux0=numpy.pi*6):
    circuit = [J(0,1,Ej,'Jx',_J_[1],_J_[0]),J(2,3,Ej,'Jy',_J_[1],_J_[0])]
    circuit += [C(0,1,EcJ,'CJx',_CJ_[1],_CJ_[0]),C(2,3,EcJ,'CJy',_CJ_[1],_CJ_[0])]
    circuit += [L(3,0,El,'Lx',True,_L_[1],_L_[0]),L(1,2,El,'Ly',True,_L_[1],_L_[0])]
    circuit += [C(3,1,Ec,'Cx',_C_[1],_C_[0]),C(0,2,Ec,'Cy',_C_[1],_C_[0])]
    pairs = dict()
    if symmetry:
        pairs['Ly'] = 'Lx'
        pairs['Cy'] = 'Cx'
        pairs['Jy'] = 'Jx'
        pairs['CJy'] = 'CJx'
    
    circuit = Circuit(circuit,basis,sparse,pairs)
    if ridge :
        circuit.R = tensor([[1,-1,-1],[-1,1,-1],[-1,-1,1]])/2.
        circuit.L_,circuit.C_ = circuit.modeTransformation()
        def ridgeHamiltonianLC(self):
            basis = self.basis
            #Ln_,Cn_ = self.Ln_,self.Cn_
            #R = self.R
            #L_ = inv(R.T) @ Ln_ @ inv(R); L_ = diag(diag(L_))
            #C_ = R @ Cn_ @ R.T; C_ = diag(diag(C_))
            #L_,C_ = self.modeTransformation()
            # Chi mode impedance
            self.Cn_,self.Ln_ = self.componentMatrix()
            self.L_,self.C_ = self.modeTransformation()
            L_,C_ = self.L_,self.C_
            z = sqrt(C_[0,0]/L_[0,0])
            
            F = [self.backend.basisFo(basis['Chi'],z)]
            F += [self.backend.basisFf(basis['Phi'],flux0)]
            F += [self.backend.basisFq(basis['Theta'])]

            Q = [self.backend.basisQo(basis['Chi'],z)]
            Q += [4*self.backend.basisFiniteII(basis['Phi'],(-flux0,flux0))]
            Q += [self.backend.basisQq(basis['Theta'])]

            H_LC = self.backend.modeMatrixProduct(F,L_,F,(0,0))/2
            # explicit basis change
            H_LC += - C_[1,1]*self.backend.basisProduct(Q,[1])/2
            H_LC += self.backend.modeMatrixProduct(Q,C_[2:,2:],Q,(2,2))/2
            H_LC += self.backend.modeMatrixProduct(Q,C_[:1,:1],Q,(0,0))/2
            
            return H_LC

        def ridgeJosephson(self):
            def Hamiltonian(flux):
                flux = flux['Lx']
                Ej = self.josephsonComponents()[1][0]
                D_Phi = self.backend.displacementFlux(basis['Phi'],flux0,1)*phase(flux/2) 
                D_Phi += self.backend.displacementFlux(basis['Phi'],flux0,-1)*phase(-flux/2) 
                D_Chi = self.backend.displacementOscillator(basis['Chi'],tensor(1.),1)
                D_Chi += self.backend.displacementOscillator(basis['Chi'],tensor(1.),-1)
                D_Theta = self.backend.chargeDisplacePlus(basis['Theta']) 
                D_Theta += self.backend.chargeDisplaceMinus(basis['Theta'])
                D = [D_Chi/2,D_Phi/2,D_Theta/2]
                H_J = -2*Ej*self.backend.basisProduct(D,[1,2])
                return H_J
            return Hamiltonian

        def kermanBasisSize():
            basis = circuit.basis
            return (2*basis['Theta']+1)*(basis['Phi'])*basis['Chi']
        circuit.ridgeJosephson = ridgeJosephson
        circuit.ridgeHamiltonianLC = ridgeHamiltonianLC
        circuit.kermanBasisSize = kermanBasisSize
    
    return circuit



def prismon(basis,Ej=10.,Ec=50.,El=10.,EcJ=100.,sparse=True,symmetry=False,_L_=(L_,L0),_C_=(C_,C0),_J_=(J_,J0),_CJ_=(4*C_,4*C0)):
    circuit =  [L(0,1,El,'La',True,_L_[1],_L_[0]),C(0,2,Ec,'Ca',_C_[1],_C_[0]),J(1,2,Ej,'Ja',_J_[1],_J_[0]),C(1,2,EcJ,'CJa',_CJ_[1],_CJ_[0])]
    circuit += [L(2,3,El,'Lb',True,_L_[1],_L_[0]),C(1,5,Ec,'Cb',_C_[1],_C_[0]),J(0,4,Ej,'Jb',_J_[1],_J_[0]),C(0,4,EcJ,'CJb',_CJ_[1],_CJ_[0])]
    circuit += [L(5,4,El,'Lc',True,_L_[1],_L_[0]),C(4,3,Ec,'Cc',_C_[1],_C_[0]),J(3,5,Ej,'Jc',_J_[1],_J_[0]),C(3,5,EcJ,'CJc',_CJ_[1],_CJ_[0])]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['Jb'] = 'Ja' ; pairs['Jc'] = 'Ja'
        pairs['Cb'] = 'Ca' ; pairs['Cc'] = 'Ca'
        pairs['Lb'] = 'La' ; pairs['Lc'] = 'La'
        pairs['CJb'] = 'CJa' ; pairs['CJc'] = 'CJa'
        
    circuit = Circuit(circuit,basis,sparse,pairs)
    return circuit

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
    oscillator = [L(0,1,El,'L'),C(0,1,Ec,'C')]
    return Circuit(oscillator,basis,sparse)

def fluxoniumArray(basis,shunt=None,gamma=1.5,N=0,Ec=100,Ej=150,sparse=True):
    # N : number of islands
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'Junc')]
    if shunt is None:
        shunt = Ec/gamma
    for i in range(N):
        circuit += [J(1+i,2+i,gamma*Ej,'junc'+str(i))]
        circuit += [C(1+i,2+i,shunt,'cap'+str(i),C_=0.)]
    circuit += [J(1+N,0,gamma*Ej,'junc'+str(N))]
    circuit += [C(1+N,0,shunt,'cap'+str(N),C_=0.)]
    
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def fluxonium(basis,El=.0003,Ec=.1,Ej=20,sparse=True):
    circuit = [C(0,1,Ec,'Cap')]
    circuit += [J(0,1,Ej,'JJ')]
    circuit += [L(0,1,El,'I',True)]

    circuit = Circuit(circuit,basis,sparse)
    return circuit

def shuntedQubit(basis,josephson=[120.,50,120.],cap=[10.,50.,10.],ind=100.,sparse=True,symmetry=False,_C_=(C_,C0),_J_=(J_,J0)):
    Ej1,Ej2,Ej3 = josephson
    C1,C2,C3 = cap

    circuit = [J(1,2,Ej1,'JJ1'),C(1,2,C1,'C1',_C_[1],_C_[0])]
    circuit += [J(2,3,Ej2,'JJ2'),C(2,3,C2,'C2',_C_[1],_C_[0])]
    circuit += [J(3,0,Ej3,'JJ3'),C(3,0,C3,'C3',_C_[1],_C_[0])]
    circuit += [L(0,1,ind,'I',True)]
    
    # inbuilt symmetry
    pairs = dict()
    if symmetry:
        pairs['JJ3'] = 'JJ1'
        pairs['C3'] = 'C1'
    
    circuit = Circuit(circuit,basis,sparse,pairs)
    return circuit

def shuntedQubitFluxFree(basis,josephson=[120.,50,120.],cap=[10.,50.,10.],sparse=True):
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

def resI(basis,sparse=True):
    cap = 54.3 + 1.92
    circuit = [C(0,1,capE(cap*1e-15),'C')] + [J(0,1,9.13,'J1')] + [J(0,1,1.00,'J2')]
    circuit += [L(0,1,.001,'L',True)]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def resIV(basis,sparse=True):
    cap = 28.0
    Ec = capE(cap*1e-15)
    El = indE(150.*1e-9)
    circuit = [C(0,1,Ec,'C')] + [J(0,1,9.13,'J')]
    circuit += [L(0,1,El,'L',True)]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def resII(basis,sparse=True):
    Ec = capE(.15*1e-15)
    Ec1 = capE(3.4*1e-12)
    Ec2 = capE(1.5*1e-12)
    circuit = [C(0,1,Ec,'C')]
    circuit += [L(0,1,.001,'L',True)]
    circuit += [C(1,2,Ec1,'C1'), J(1,2,100.,'J1')]
    circuit += [C(2,0,Ec2,'C2'), J(2,0,88.2,'J2')]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def resV(basis,sparse=True):
    Ec = capE(45.*1e-15)
    El = indE(19.8*1e-9)
    EcJ = capE(15.0*1e-15)
    circuit = [C(0,1,Ec,'C')]
    circuit += [L(1,2,.001,'L',True)]
    circuit += [C(2,0,EcJ,'EcJ'), J(2,0,88.2,'J')]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def resIII(basis,sparse=True):
    Ec = capE(101.*1e-15)
    El = indE((150.+18.3)*1e-9)
    EcLink = capE(.8*1e-12)
    Ec1 = capE(1.59*1e-12)
    Ec2 = capE(520*1e-12)
    
    circuit = [C(0,1,Ec,'C')]
    circuit += [L(0,2,El,'L',True)]
    circuit += [C(1,2,Ec1,'C1'), J(1,2,1.,'J1')]
    circuit += [C(2,3,EcLink,'Coupling')]
    circuit += [C(3,0,Ec2,'C2'), J(3,0,59.5,'J2')]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

def resVI(basis,sparse=True):
    Ec1 = capE(.15*1e-15)
    El = indE((150.+18.3)*1e-9)
    EcLink = capE(.8*1e-12)
    Ec1 = capE(1.59*1e-12)
    Ec2 = capE(520*1e-12)
    
    circuit = [C(0,1,Ec,'C1')]
    circuit += [L(0,2,El,'L1',True)]
    circuit += [C(1,2,Ec1,'C1'), J(1,2,1.,'J1')]
    circuit += [C(2,3,EcLink,'Coupling')]
    circuit += [C(3,0,Ec2,'C2'), J(3,0,59.5,'J2')]
    circuit = Circuit(circuit,basis,sparse)
    return circuit

if __name__=='__main__':
    circuit = shuntedQubit()
    print(circuit.modeDistribution())
    circuit.basis = {'O':[10],'I':[0],'J':[10,10]}
    H_LC = circuit.kermanHamiltonianLC()
    H_J = circuit.kermanHamiltonianJosephson({'I':tensor(.25)})

    import ipdb; ipdb.set_trace()
