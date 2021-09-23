import networkx,copy
import matplotlib.pyplot as plt
from components import *
from numpy.linalg import det
from numpy import kron

def modeTensorProduct(pre,M,post):
    """
        extend mode to full system basis
        sequentially process duplication
    """
    H = 1
    for dim in pre:
        H = numpy.kron(H,numpy.identity(dim))
    H = numpy.kron(H,M)
    for dim in post:
        H = numpy.kron(H,numpy.identity(dim))
    return H

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
        M : mode operator, implementing mode interactions
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
                left = numpy.kron(left,numpy.identity(1,len(right)))
                right = numpy.kron(numpy.identity(len(left),1),right)

            H += M[i,j]*dotProduct(left,right)

    return H

def dotProduct(left,right):
    return numpy.dot(left,right)

def inverse(A):
    if numpy.linalg.det(A) == 0:
        return numpy.zeros_like(A)
    return numpy.linalg.inv(A)

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def transformation(M,T):
    return dotProduct(T.T,dotProduct(M,T))

def hamiltonianEnergy(H):
    eigenenergies = numpy.real(numpy.linalg.eigvals(H))
    eigenenergies.sort()
    return eigenenergies

class Circuit:
    """
        * no external fluxes must be in parallel : redundant
        * no LC component must be in parallel : redundant
        * only Josephson elements allowed in parallel
    """
    """
        * u,v : indices of raw graph : GL, minimum_spanning_tree
        * i,j : indices of indexed graph : mode correspondence
    """

    def __init__(self,network,basis):
        self.network = network
        self.G = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.basis = basis
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.edges_inductive = self.edgesInductance()
        self.Nb = len(self.edges_inductive)
        self.Cn_,self.Ln_ = self.componentMatrix()

    def parseCircuit(self):
        G = networkx.MultiGraph()
        for component in self.network:
            weight = 1
            if component.__class__ == J:
                weight = component.energy * 1e-9
            G.add_edge(component.plus,component.minus,key=component.ID,weight=weight,component=component)
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
        edges = dict([*enumerate([*self.G.edges(keys=True)])])
        return edges

    def spanningTree(self):
        GL = self.graphGL()
        S = networkx.minimum_spanning_tree(GL)
        return S

    def graphGL(self):
        GL = copy.deepcopy(self.G)
        capacitance_edges = []
        for u,v,component in GL.edges(data=True):
            component = component['component']
            if component.__class__ == C:
                capacitance_edges.append((u,v,component.ID))
        GL.remove_edges_from(capacitance_edges)

        return GL

    def componentMatrix(self):
        Cn = self.nodeCapacitance()
        Cn_ = inverse(Cn)
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        Ln_ = transformation(inverse(Lb+M),Rbn)

        return Cn_,Ln_

    def loopFlux(self,u,v,key,external_fluxes):
        """
            external_fluxes : {identifier:flux_value}
        """
        flux = 0
        external = set(external_fluxes.keys())
        S = self.spanning_tree
        path = networkx.shortest_path(S,u,v)
        for i in range(len(path)-1):
            multi = S.get_edge_data(path[i],path[i+1])
            match = external.intersection(set(multi.keys()))
            # multiple external flux forbidden on same branch
            assert len(match) <= 2
            if len(match)==1 :
                component = multi[match.pop()]['component']
                assert component.__class__ == L
                assert component.external == True
                flux += external_fluxes[component.ID]

        return flux

    def josephsonComponents(self):
        edges,Ej = [],[]
        for index,(u,v,key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == J:
                edges.append((u,v,key))
                Ej.append(component.energy)
        return edges,Ej

    def fluxBiasComponents(self):
        """
            Inducer : Inductor introduces external flux bias
        """
        edges,L_ext = [],[]
        for index,(u,v,key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                if component.external:
                    edges.append((u,v,key))
                    L_ext.append(component.inductance)
        return edges,L_ext

    def edgesInductance(self):
        edges_inductive = dict()
        for index,(u,v,key) in self.edges.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                edges_inductive[index] = (u,v,key)
        return edges_inductive

    def nodeCapacitance(self):
        Cn = numpy.zeros((self.Nn,self.Nn))
        for i,node in self.nodes.items():
            for u,v,component in self.G.edges(node,data=True):
                component = component['component']
                if component.__class__ == C:
                    capacitance = component.capacitance
                    Cn[i,i] += capacitance
                    if not (u==0 or v==0):
                        Cn[self.nodes_[u],self.nodes_[v]] = -capacitance
                        Cn[self.nodes_[v],self.nodes_[u]] = -capacitance
        return Cn

    def branchInductance(self):
        Lb = numpy.zeros((self.Nb,self.Nb))
        #numpy.fill_diagonal(Lb,L_limit)
        for index,(u,v,key) in self.edges_inductive.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                #if not component.external:
                Lb[index,index] = component.inductance
        return Lb

    def mutualInductance(self):
        M = numpy.zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = numpy.zeros((self.Nb,self.Nn),int)
        for index,(u,v,key) in self.edges_inductive.items():
            if not u==0:
                Rbn[index][self.nodes_[u]] = 1
            if not v==0:
                Rbn[index][self.nodes_[v]] = -1

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

    def hamiltonianKerman(self,basis):
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

    def josephsonEnergy(self,external_fluxes=dict()):
        basis = self.basis
        Dplus = [chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        H = 0
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            #print(u,v,i,j,flux)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus = basisProduct(Dplus,[i])
                Jminus = basisProduct(Dminus,[i])
            else:
                Jplus = crossBasisProduct(Dplus,Dminus,i,j)
                Jminus = crossBasisProduct(Dplus,Dminus,j,i)
                #assert (Jminus == Jplus.conj().T).all()
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H -= Hj

        return H/2

    def fluxInducerEnergy(self):
        basis = self.basis
        fluxModes = [basisPj(basis_max) for basis_max in basis]
        edges,L_ext = self.fluxBiasComponents()
        H = 0
        for (u,v,key),L in zip(edges,L_ext):
            i,j = self.nodes_[u],self.nodes_[v]
            #print(u,v,i,j,L)
            if i<0 or j<0 :
                # grounded inducer
                i = max(i,j)
                P = basisProduct(fluxModes,[i])
            else:
                P = basisProduct(fluxModes,[i]) - basisProduct(fluxModes,[j])
            H += dotProduct(P,P) / 2 / L
        return H

    def oscillatorHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        Q = [basisQo(2*basis_max+1,impedance[index]) for index,basis_max in enumerate(basis)]
        P = [basisPo(2*basis_max+1,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(P,Ln_,P)

        H = (H_C+H_L)/2

        return H

    def fluxHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        Q = [chargeFlux(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        P = [fluxFlux(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(P,Ln_,P)

        H = (H_C+H_L)/2

        return H

    def chargeHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        Q = [chargeCharge(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        P = [fluxCharge(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(P,Ln_,P)

        H = (H_C+H_L)/2

        return H

    def kermanHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [basisQji(basis_max) for basis_max in basis]
        P = [basisPj(basis_max) for basis_max in basis]
        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(P,Ln_,P)

        H = (H_C+H_L)/2

        return H

    def circuitEnergy(self,H_LC=0,external_fluxes=dict(),exclude=False):
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        H_J = self.josephsonEnergy(external_fluxes)
        H = H_LC + H_J
        if exclude:
            H = H[:-1,:-1]
        eigenenergies = hamiltonianEnergy(H)
        return eigenenergies

    def spectrumManifold(self,flux_points,flux_manifold,H_LC=0,excitation=1):
        """
            flux_points : inductor identifier for external introduction
            flux_manifold : [(fluxes)]
        """
        #manifold of flux space M
        energy_spectrum,E0 = [],[]
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        for fluxes in flux_manifold:
            external_fluxes = dict(zip(flux_points,fluxes))
            H_J = self.josephsonEnergy(external_fluxes)
            H = H_LC + H_J
            eigenenergies = hamiltonianEnergy(H)
            E0.append(eigenenergies[0])
            energy_spectrum.append(eigenenergies[excitation]-eigenenergies[0])
        return E0,energy_spectrum

if __name__=='__main__':
    circuit = shuntedQubit([2,2,2])
    utils.plotMatPlotGraph(circuit.G,'circuit')
    utils.plotMatPlotGraph(circuit.spanning_tree,'spanning_tree')
    flux_manifold = zip(numpy.arange(0,1,.01))
    from test_flux_spectrum import josephsonE
    Ej0 = josephsonE(1,.125)
    Ej1 = circuit.josephsonEnergy({'I':.125})
    e1 = hamiltonianEnergy(Ej1)
    Ej2 = circuit.josephsonEnergy({'I':.0003})
    e2 = hamiltonianEnergy(Ej2)
    import ipdb; ipdb.set_trace()
