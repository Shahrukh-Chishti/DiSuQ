import numpy,networkx
import matplotlib.pyplot as plt
from components import *
import numpy
#import jax.numpy as jnp
#from jax import grad, jit
#from jax import random

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

def crossBasisProduct(A,B,a,b):
    assert len(A)==len(B)
    n = len(A)
    product = 1
    for i in range(n):
        if i==a:
            product = numpy.kron(product,A[i])
        elif i==b:
            product = numpy.kron(product,B[i])
        else:
            product = numpy.kron(product,numpy.identity(len(A[i])))
    return product

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
    if numpy.linalg.det(A) == 0:
        return numpy.zeros_like(A)
    return numpy.linalg.inv(A)

def phase(phi):
    return exp(im*2*pi*phi)

def transformation(M,T):
    # need reecheck for transpose
    return numpy.dot(T.T,M.dot(T))

class Circuit:
    """
        * no external fluxes must be in parallel : redundant
        * no LC component must be in parallel : redundant
        * only Josephson elements allowed in parallel
    """

    def __init__(self,network,basis):
        self.network = network
        self.G = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.basis = basis
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Nb = len(self.edges)
        self.Cn_,self.Ln_ = self.componentMatrix()

    def parseCircuit(self):
        G = networkx.MultiGraph()
        for u,v,component in self.network:
            weight = 1
            if component.__class__ == J:
                weight = component.energy
            G.add_edge(u,v,weight=weight,component=component)
        return G

    def nodeIndex(self):
        # nodes identify variable placeholders
        nodes = list(self.G.nodes())
        nodes.remove(0) # removing ground from active nodes
        nodes = dict([*enumerate(nodes)])

        nodes_ = {val:key for key,val in nodes.items()}
        nodes_[0] = -1
        return nodes,nodes_

    def edgesIndex(self):
        """
            {(u,v):[,,,]} : MultiGraph edges
        """
        edges = dict([*enumerate([*self.G.edges()])])
        return edges

    def componentMatrix(self):
        Cn_ = self.nodeCapacitance()
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        Ln_ = transformation(inverse(Lb+M),Rbn)

        return Cn_,Ln_

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

    def hamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [basisQji(basis_max) for basis_max in basis]
        P = [basisPj(basis_max) for basis_max in basis]
        C = modeMatrixProduct(Q,Cn_,Q)
        L = modeMatrixProduct(P,Ln_,P)

        H = (C+L)/2

        return H

    def spectrum(self,flux_range,flux_points):
        """
            external_fluxes : [edges index: range]
        """
        #manifold of flux space M
        energy_spectrum = []
        for point in manifold:
            Hj = self.josephsonEnergy(point)
            H = Hj + self.hamiltonianLC()
            eigenenergies = numpy.linalg.eigvals(H)
            energy_spectrum.append(eigenenergies.sort()[0])
        return energy_spectrum

    def loopFlux(self,u,v,external_fluxes):
        flux = 0
        S = self.spanning_tree
        path = networkx.shortest_path(S,u,v)
        for i in range(len(path)-1):
            component = S.get_edge_data(path[i],path[i+1])['component']
            if component.__class__ == L:
                if not component.external is None:
                    flux += external_fluxes[component.external]

        return flux

    def josephsonEnergy(self,indices,external_fluxes):
        """
            external_fluxes : {identifier:flux_value}
        """
        basis = self.basis
        Dplus = [chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        assert len(external_fluxes) == len(edges)

        Hj = 0*im
        for edge,E in zip(edges,Ej):
            i,j = edge # assuming polarity on nodes
            i,j = self.nodes_[i],self.nodes_[j]
            if i<0 or j<0:
                # grounded josephson, without external flux
                i = max(i,j)
                Jplus = E*basisProduct(Dplus,[i])*phase(flux)
                Jminus = E*basisProduct(Dminus,[i])*phase(-flux)
            else:
                Jplus = E*crossBasisProduct(Dplus,Dminus,i,j)*phase(flux)
                Jminus = E*crossBasisProduct(Dplus,Dminus,j,i)*phase(-flux)
            Hj -= Jplus + Jminus

        return Hj/2

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
                component = component['component']
                if component.__class__ == C:
                    C = component.capacitance
                    Cn[i,i] += C
                    if not (u==0 or v==0):
                        Cn[self.nodes_[u],self.nodes_[v]] = -C
                        Cn[self.nodes_[v],self.nodes_[u]] = -C

        Cn_ = inverse(Cn)
        return Cn_

    def branchInductance(self):
        Lb = numpy.zeros((self.Nb,self.Nb))
        for i,(u,v) in self.edges.items():
            component = self.G[u][v]
            component = component['component']
            if component.__class__ == L:
                Lb[i,i] = component['L'].inductance
        return Lb

    def mutualInductance(self):
        M = numpy.zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = numpy.zeros((self.Nb,self.Nn),int)
        for i,(u,v) in self.edges.items():
            if not (u==0 or v==0):
                # no polarity on ground
                Rbn[i][self.nodes_[u]] = 1
                Rbn[i][self.nodes_[v]] = -1
        return Rbn

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

    def modeTransformation(self,Ln_):
        No,Ni,Nj = self.modeDistribution(Ln_)
        return R,No,Ni,Nj

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

if __name__=='__main__':
    H = transmon.hamiltonian_charged([5])
    spectrum = numpy.linalg.eigvals(H)
    spectrum.sort()
    #networkx.draw_spring(transmon.G)
    #plt.show()
    import ipdb;ipdb.set_trace()
