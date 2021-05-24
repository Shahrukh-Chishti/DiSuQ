import numpy,networkx
from components import J,L,C
import numpy
import jax.numpy as jnp
from jax import grad, jit
from jax import random

flux_quanta = 1
h = 1

def modeTensorProduct():
    """
        extend mode to full system basis
    """

def modeQuantProduct(A,M,B):
    """
        M : N x N : mode operator
        B : basisB x basisB : basis operator yielding N-tuple
        A : basisA x basisA : basis operator(transpose) yield N-tuple
        returns : basisA x basisB mode state Hamiltonian matrix
    """

def inverse(A):
    return numpy.linalg.inv(A)

def transformation(M,T):
    return numpy.dot(T.t,M.dot(T))

class Circuit:
    def __init__(self,network):
        self.network = network
        self.G = self.parseCircuit()
        self.nodes = self.nodeIndex()
        self.edges = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Nb = len(self.edges)

    def parseCircuit(self):
        G = networkx.Graph()
        for component in self.network:
            G.add_edge(component.minus,component.plus,component=component)
        return G

    def transformComponents(self):
        Cn_ = self.nodeCapacitance()
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()

        Ln_ = transformation(inverse(Lb+M),Rbn)
        R,N0,Ni,Nj = self.modeTransformation(Ln_)
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

    def hamiltonian(self,basis):
        """
            basis : {o:(,,,),i:(,,,),j:(,,,)}
        """
        Lo_,C_ = self.transformComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_

        Zi = numpy.sqrt(numpy.diagonal(Co_)/numpy.diagonal(Lo_))

    def modeDistribution(self):
        return No,Ni,Nj

    def spanningTree(self):
        GL = self.graphGL()
        S = networkx.minimum_spanning_tree(GL)

    def graphGL(self):
        GL = self.G.copy()
        for u,v,component in GL.edges(data=True):
            component = component['component']
            if component.__class__ == C:
                GL.remove_edge(u,v)
        return GL

    def nodeIndex(self):
        nodes = dict([*enumerate([*self.G.nodes()])])
        return nodes

    def edgesIndex(self):
        edges = dict([*enumerate([*self.G.edges()])])
        return edges

    def nodeCapacitance(self):
        Cn = numpy.zeros((self.Nn,self.Nn))
        for i,node in self.nodes.items():
            node_capacitance = 0
            for u,v,component in self.G.edges(node,data=True):
                component = component['component']
                if component.__class__ == C:
                    node_capacitance += component.capacitance
            Cn[i,i] = node_capacitance
            # off diagonals path non-uniqueness

        Cn_ = inverse(Cn)
        return Cn_

    def branchInductance(self):
        Lb = numpy.zeros((self.Nb,self.Nb))
        for i,(u,v) in self.edges.items():
            component = self.G[u][v]
            component = component['component']
            if component.__class__ == L:
                Lb[i,i] = component.inductance
        return Lb

    def mutualInductance(self):
        M = numpy.zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = numpy.zeros((self.Nb,self.Nn),int)
        node_ = {val:key for key,val in self.nodes.items()}
        for i,(u,v) in self.edges:
            Rbn[i][node_[u]] = 1
            Rbn[i][node_[v]] = -1
        return Rbn

    def modeTransformation(self,Ln_):
        return R

if __name__=='__main__':
    transmon = [J(0,1,10),C(0,1,5)]
    transmon = Circuit(transmon)
    import ipdb;ipdb.set_trace()
