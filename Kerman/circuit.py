import numpy,networkx
import matplotlib.pyplot as plt
from components import *
import numpy
import jax.numpy as jnp
from jax import grad, jit
from jax import random

def modeTensorProduct(pre,M,post):
    """
        extend mode to full system basis
        sequentially process duplication
    """
    H = 1
    for dim in pre:
        H = numpy.kron(H,numpy.ones(dim))
    H = numpy.kron(H,M)
    for dim in post:
        H = numpy.kron(H,numpy.ones(dim))
    return H

def basisProduct(O,index):
    """
        O : basis_operators : list of quantised operators
        index : index of basis representation
        returns B : full basis representation : n_basis X n_basis
    """
    n = len(O)
    B = 1
    for i in range(index-1):
        n_basis = len(O[i])
        B = numpy.kron(B,numpy.ones(n_basis,n_basis))
    B = numpy.kron(B,O[index])
    for i in range(index,n):
        n_basis = len(O[i])
        B = numpy.kron(B,numpy.ones(n_basis,n_basis))
    return B

def basisProduct(O,indices):
    n = len(O)
    B = 1
    for i in range(n):
        if i in indices:
            B = numpy.kron(B,O[i])
        else:
            B = numpy.kron(B,numpy.identity(len(O[i])))
    return B

def modeMatrixProduct(A,M,B,cross_product=False):
    """
        M : mode operator
        B : list : basis operators
        A : list : basis operators(transpose)
        cross_product : indicates if A!=B, assumed ordering : AxB
        returns : prod(nA) x prod(nB) mode state Hamiltonian matrix
    """
    H = 0
    nA,nB = len(A),len(B)
    assert M.shape==(nA,nB)
    for i in range(nA):
        for j in range(nB):
            left = basisProduct(A,[i])
            right = basisProduct(B,[j])
            if cross_product:
                left = numpy.kron(left,numpy.ones(1,len(right)))
                right = numpy.kron(numpy.ones(len(left),1),right)

            H += M[i,j]*numpy.dot(left,right)

    return H

def inverse(A):
    return numpy.linalg.inv(A)

def transformation(M,T):
    # need reecheck for transpose
    return numpy.dot(T.T,M.dot(T))

class Circuit:
    def __init__(self,network):
        self.network = network
        self.G = self.parseCircuit()
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Nb = len(self.edges)
        self.Cn_,self.Ln_ = self.componentMatrix()

    def parseCircuit(self):
        G = networkx.Graph()
        for u,v,components in self.network:
            G.add_edge(u,v,**components)
        return G

    def componentMatrix(self):
        Cn_ = self.nodeCapacitance()
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        Ln_ = transformation(inverse(Lb+M),Rbn)

        return Cn_,Ln_

    def transformComponents(self):
        Cn_,Ln_ = self.componentMatrix()

        R,No,Ni,Nj = self.modeTransformation(Ln_)
        L_ = transformation(Ln_,inverse(R))

        Lo_ = L_[:No,:No]
        C_ = transformation(Cn_,R.t)
        Co_ = C_[:No,:No]
        Coi_ = C_[:No,No:-Nj]
        Coj_ = C_[:No,-Nj:]
        Ci_ = C_[No:-Nj,No:-Nj]
        Cij_ = C_[No:-Nj,-Nj:]
        Cj_ = C_[-Nj:,-Nj:]

        C_ = Co_,Coi_,Coj_,Ci_,Cij_,Cj_

        return Lo_,C_

    def hamiltonian_charged(self,basis):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_

        Q = [basisQji(basis_max) for basis_max in basis]
        P = [basisPj(basis_max) for basis_max in basis]
        Dplus = [chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis] 
        C = modeMatrixProduct(Q,Cn_,Q)
        L = modeMatrixProduct(P,Ln_,P)

        Hlc = (C+L)/2
        Hj = 0
        edges,Ej = self.josephsonComponents()

        for edge,E in zip(edges,Ej):
            i,j = edge # assuming polarity on nodes
            Jplus = basisProduct(Dplus,[i]) + basisProduct(Dminus,[i])
            Jminus = basisProduct(Dplus,[j]) + basisProduct(Dminus,[j])
            Hj += E*(1-Jplus+Jminus)/2

        H = Hlc + Hj
        return H

    def hamiltonian(self,basis):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        Lo_,C_ = self.transformComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        n_baseO,n_baseI,n_baseJ = self.modeBasisSize(basis)

        Z = numpy.sqrt(numpy.diagonal(Co_)/numpy.diagonal(Lo_))
        Qo = [basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [basisQji(basis_max) for basis_max in basis['I']]
        Qj = [basisQji(basis_max) for basis_max in basis['J']]

        Co = modeMatrixProduct(Qo,Co_,Qo)
        Co = modeTensorProduct(((n_baseJ,n_baseJ),(n_baseI,n_baseI)),Co,(1,1))

        Po = [basisPo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Lo = modeMatrixProduct(Po,Lo_,Po)
        Lo = modeTensorProduct(((n_baseJ,n_baseJ),(n_baseI,n_baseI)),Lo,(1,1))

        Ho = Co + Lo + Uo

        Coi = modeMatrixProduct(Qo,Coi_,delQi)
        Coi = modeTensorProduct()

        Coj = modeMatrixProduct(Qo,Coj_,delQj)
        Coj = modeTensorProduct()

        Cij = modeMatrixProduct(delQi,Cij_,delQj)
        Cij = modeTensorProduct()

        Hint = Coi + Coj + Cij + Uint

        Ci = modeMatrixProduct(delQi,Ci_,delQi)
        Ci = modeTensorProduct()

        Hi = Ci

        Cj = modeMatrixProduct(delQj,Ci_delQj)
        Cj = modeTensorProduct()

        Hj = Cj + Uj

        return Ho+Hint+Hi+Hj

    def modeBasisSize(self,basis):
        n_baseO *= numpy.prod(basis['O'])
        n_baseI *= numpy.prod(basis['I'])
        n_baseJ *= numpy.prod(basis['J'])

        return n_baseO,n_baseI,n_baseJ

    def modeDistribution(self,Ln_):
        Ni = 1 # defaulted
        No = numpy.linalg.matrix_rank(Ln_)
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def spanningTree(self):
        GL = self.graphGL()
        S = networkx.minimum_spanning_tree(GL)
        return S

    def graphGL(self):
        GL = self.G.copy()
        for u,v,component in GL.edges(data=True):
            component = component['component']
            if component.__class__ == C:
                GL.remove_edge(u,v)
        return GL

    def nodeIndex(self):
        nodes = list(self.G.nodes())
        nodes.remove(0) # removing ground from active nodes
        nodes = dict([*enumerate(nodes)])

        nodes_ = {val:key for key,val in nodes.items()}
        nodes_[0] = 0
        return nodes,nodes_

    def edgesIndex(self):
        edges = dict([*enumerate([*self.G.edges()])])
        return edges

    def josephsonComponents(self):
        edges,Ej = [],[]
        for i,(u,v) in self.edges.items():
            component = self.G[u][v]
            if 'J' in component:
                edges.append((u,v))
                Ej.append(component['J'].energy)
        return edges,Ej

    def nodeCapacitance(self):
        Cn = numpy.zeros((self.Nn,self.Nn))
        for i,node in self.nodes.items():
            for u,v,component in self.G.edges(node,data=True):
                #component = component['component']
                #if component.__class__ == C:
                if 'C' in component:
                    C = component['C'].capacitance
                    Cn[i,i] += C
                    Cn[self.nodes_[u],self.nodes_[v]] = -C
                    Cn[self.nodes_[v],self.nodes_[u]] = -C

        Cn_ = inverse(Cn)
        return Cn_

    def branchInductance(self):
        Lb = numpy.zeros((self.Nb,self.Nb))
        for i,(u,v) in self.edges.items():
            component = self.G[u][v]
            #component = component['component']
            #if component.__class__ == L:
            if 'L' in component:
                Lb[i,i] = component['L'].inductance
        return Lb

    def mutualInductance(self):
        M = numpy.zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = numpy.zeros((self.Nb,self.Nn),int)
        for i,(u,v) in self.edges.items():
            if not u==0 or v==0:
                # no polarity on ground
                Rbn[i][self.nodes_[u]] = 1
                Rbn[i][self.nodes_[v]] = -1
        return Rbn

    def modeTransformation(self,Ln_):
        No,Ni,Nj = self.modeDistribution(Ln_)
        return R,No,Ni,Nj

if __name__=='__main__':
    transmon = [[0,1,{'J':J(0,1,10),'C':C(0,1,5),'L':L(0,1,100)}]]
    transmon = Circuit(transmon)
    H = transmon.hamiltonian_charged([5])
    spectrum = numpy.linalg.eigvals(H)
    spectrum.sort()
    #networkx.draw_spring(transmon.G)
    #plt.show()
    import ipdb;ipdb.set_trace()
