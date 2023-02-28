import networkx,copy
import matplotlib.pyplot as plt
from numpy import kron,full
from .components import *

def modeTensorProduct(pre,M,post):
    """
        extend mode to full system basis
        sequentially process duplication
    """
    H = 1
    for dim in pre:
        H = kron(H,identity(dim))
    H = kron(H,M)
    for dim in post:
        H = kron(H,identity(dim))
    return H

def crossBasisProduct(A,B,a,b):
    assert len(A)==len(B)
    n = len(A)
    product = 1
    for i in range(n):
        if i==a:
            product = kron(product,A[i])
        elif i==b:
            product = kron(product,B[i])
        else:
            product = kron(product,identity(len(A[i])))
    return product

def basisProduct(O,indices=None):
    # indices :: modes
    n = len(O)
    B = 1
    if indices is None:
        indices = arange(n)
    for i in range(n):
        if i in indices:
            B = kron(B,O[i])
        else:
            B = kron(B,identity(len(O[i])))
    return B

def modeMatrixProduct(A,M,B,mode=(0,0)):
    """
        M : mode operator, implementing mode interactions
        A,B : complete list of basis operators
        mode : demarcates subset of interacting mode
    """
    H = 0
    N = len(A)
    assert len(A)==len(B)
    nA,nB = M.shape
    a,b = mode
    for i in range(nA):
        for j in range(nB):
            left = basisProduct(A,[i+a])
            right = basisProduct(B,[j+b])
            #if cross_mode:
            #    left = kron(left,identity(len(right)))
            #    right = kron(identity(len(left)),right)

            H += M[i,j]*dotProduct(left,right)

    return H

def dotProduct(left,right):
    return dot(left,right)

def inverse(A):
    if det(A) == 0:
        return zeros_like(A)
    return inv(A)

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def hamiltonianEnergy(H,sort=True):
    eigenenergies = real(eigvalsh(H))
    if sort:
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
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges,self.edges_inductive = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nb = len(self.edges_inductive)

        self.Cn_,self.Ln_ = self.componentMatrix()
        self.No,self.Ni,self.Nj = self.modeDistribution()
        self.R = self.modeTransformation()
        self.Lo_,self.C_ = self.transformComponents()

        self.basis = basis
        # basis : list of basis_size of ith mode
        # basis : dict of O,I,J : list of basis_size

    def parseCircuit(self):
        G = networkx.MultiGraph()
        for component in self.network:
            weight = 1e-3
            if component.__class__ == J:
                weight = component.energy
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
            inductive edges ... josephson/capactive edges
        """
        edges_G = self.G.edges(keys=True)
        index_plus,index_minus = 0, len(edges_G)-1
        edges,edges_inductive = dict(), dict()
        for u,v,key in edges_G:
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                edges_inductive[index_plus] = (u,v,key)
                edges[index_plus] = (u,v,key)
                index_plus += 1
            else:
                edges[index_minus] = (u,v,key)
                index_minus -= 1
        return edges,edges_inductive

    def spanningTree(self):
        GL = self.graphGL()
        S = networkx.minimum_spanning_tree(GL)
        return S

    def graphGL(self,elements=[C]):
        GL = copy.deepcopy(self.G)
        edges = []
        for u,v,component in GL.edges(data=True):
            component = component['component']
            if component.__class__ in elements:
                edges.append((u,v,component.ID))
        GL.remove_edges_from(edges)

        return GL

    def componentMatrix(self):
        Cn = self.nodeCapacitance()
        Cn_ = inverse(Cn)
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        Ln_ = unitaryTransformation(inverse(Lb+M),Rbn)

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

    def nodeCapacitance(self):
        Cn = zeros((self.Nn,self.Nn))
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
        Lb = zeros((self.Nb,self.Nb))
        #fill_diagonal(Lb,L_limit)
        for index,(u,v,key) in self.edges_inductive.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                #if not component.external:
                Lb[index,index] = component.inductance
        return Lb

    def mutualInductance(self):
        M = zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = zeros((self.Nb,self.Nn),int)
        for index,(u,v,key) in self.edges_inductive.items():
            if not u==0:
                Rbn[index][self.nodes_[u]] = 1
            if not v==0:
                Rbn[index][self.nodes_[v]] = -1

        return Rbn

    def modeBasisSize(self,basis):
        n_baseO = prod(array(basis['O']))
        n_baseI = prod(2*array(basis['I'])+1)
        n_baseJ = prod(2*array(basis['J'])+1)

        return n_baseO,n_baseI,n_baseJ

    def islandModes(self):
        islands = self.graphGL(elements=[C])
        islands = networkx.connected_components(islands)
        islands = list(islands)
        Ni = 0
        for sub in islands:
            if 0 not in sub:
                Ni += 1
        return Ni

    def modeDistribution(self):
        Ln_ = self.Ln_
        Ni = self.islandModes()
        No = matrix_rank(Ln_)
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def modeTransformation(self):
        Ln_ = self.Ln_
        R = diagonalisation(Ln_,True)
        return R

    def transformComponents(self):
        Cn_,Ln_ = self.componentMatrix()

        R = self.R
        No,Ni,Nj = self.No,self.Ni,self.Nj
        N = self.Nn
        L_ = unitaryTransformation(Ln_,R)

        Lo_ = L_[:No,:No]
        C_ = unitaryTransformation(Cn_,R.conj().T)
        Co_ = C_[:No,:No]
        Coi_ = C_[:No,No:No+Ni]
        Coj_ = C_[:No,No+Ni:]
        Ci_ = C_[No:No+Ni,No:No+Ni]
        Cij_ = C_[No:No+Ni,No+Ni:]
        Cj_ = C_[No+Ni:,No+Ni:]

        C_ = Co_,Coi_,Coj_,Ci_,Cij_,Cj_

        return Lo_,C_

    def modeImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        return impedance

    def oscillatorImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis['O']))]
        return impedance

    def linearCombination(self,index):
        invR = self.R.conj().T
        combination = invR[index]
        assert len(combination) == self.Nn
        return combination

    def displacementCombination(self,combination):
        basis = self.basis
        No,Ni,Nj = self.No,self.Ni,self.Nj
        O = combination[:No]
        I = combination[No:No+Ni]
        J = combination[No+Ni:]
        #assert I==0
        #assert J==1 or 0

        Z = self.oscillatorImpedance() * 2 # cooper pair factor

        DO_plus = [displacementOscillator(basis_max,z,o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DO_minus = [displacementOscillator(basis_max,z,-o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DI_plus = [displacementCharge(basis_max,i) for i,basis_max in zip(I,basis['I'])]
        DI_minus = [displacementCharge(basis_max,-i) for i,basis_max in zip(I,basis['I'])]
        DJ_plus = [displacementCharge(basis_max,j) for j,basis_max in zip(J,basis['J'])]
        DJ_minus = [displacementCharge(basis_max,-j) for j,basis_max in zip(J,basis['J'])]

        Dplus = DO_plus+DI_plus+DJ_plus
        Dminus = DO_minus+DI_minus+DJ_minus
        assert len(combination)==len(Dplus)
        assert len(combination)==len(Dminus)
        return Dplus,Dminus

    def kermanHamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        H = 0
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                combination = self.linearCombination(i)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            else:
                combination = self.linearCombination(i) - self.linearCombination(j)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = basisProduct(Dplus)
                Jminus = basisProduct(Dminus)
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H -= Hj

        return H/2

    def kermanHamiltonianLC(self):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        basis = self.basis
        Lo_ = self.Lo_
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = self.C_
        n_baseO,n_baseI,n_baseJ = self.modeBasisSize(basis)
        No,Ni,Nj = self.No,self.Ni,self.Nj

        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Qo = [basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [basisQq(basis_max) for basis_max in basis['I']]
        Qj = [basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        Co = modeMatrixProduct(Q,Co_,Q,(0,0))

        Fo = [basisFo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Fi = [basisFq(basis_max) for basis_max in basis['I']]
        Fj = [basisFq(basis_max) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        Lo = modeMatrixProduct(F,Lo_,F,(0,0))

        Ho = (Co + Lo)/2

        Coi = modeMatrixProduct(Q,Coi_,Q,(0,No))

        Coj = modeMatrixProduct(Q,Coj_,Q,(0,No+Ni))

        Cij = modeMatrixProduct(Q,Cij_,Q,(No,No+Ni))

        Hint = Coi + Coj + Cij

        Ci = modeMatrixProduct(Q,Ci_,Q,(No,No))
        Hi = Ci/2

        Cj = modeMatrixProduct(Q,Cj_,Q,(No+Ni,No+Ni))
        Hj = Cj/2

        return Ho+Hint+Hi+Hj

    def josephsonFlux(self,external_fluxes=dict()):
        basis = self.basis
        Z = self.modeImpedance() * 2 # cooper pair factor
        Dplus = [displacementFlux(basis_max,1) for z,basis_max in zip(Z,basis)]
        Dminus = [displacementFlux(basis_max,-1) for z,basis_max in zip(Z,basis)]
        edges,Ej = self.josephsonComponents()
        H = 0
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
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

    def josephsonOscillator(self,external_fluxes=dict()):
        basis = self.basis
        Z = self.modeImpedance() * 2 # cooper pair factor
        Dplus = [displacementOscillator(basis_max,z,1) for z,basis_max in zip(Z,basis)]
        Dminus = [displacementOscillator(basis_max,z,-1) for z,basis_max in zip(Z,basis)]
        edges,Ej = self.josephsonComponents()
        H = 0
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
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

    def josephsonCharge(self,external_fluxes=dict()):
        basis = self.basis
        Dplus = [chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        H = 0
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
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

        impedance = self.modeImpedance()
        Q = [basisQo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        F = [basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(F,Ln_,F)

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
        F = [fluxFlux(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H

    def fluxHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [basisQf(basis_max) for basis_max in basis]
        F = [basisFf(basis_max) for basis_max in basis]
        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(F,Ln_,F)

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
        F = [fluxCharge(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H

    def chargeHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [basisQq(basis_max) for basis_max in basis]
        F = [basisFq(basis_max) for basis_max in basis]
        H_C = modeMatrixProduct(Q,Cn_,Q)
        H_L = modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H

    def potentialOscillator(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        F = [basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        H_L = modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonOscillator(external_fluxes)
        return H_L+H_J

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [basisFq(basis_max) for basis_max in basis]
        H_L = modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonCharge(external_fluxes)
        return H_L+H_J

    def potentialFlux(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [basisFf(basis_max) for basis_max in basis]
        H_L = modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonFlux(external_fluxes)
        return H_L+H_J

    def circuitEnergy(self,H_LC=0,H_J=null,external_fluxes=dict(),exclude=False):
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        H = H_LC + H_J(external_fluxes)
        if exclude:
            H = H[:-1,:-1]
        eigenenergies = hamiltonianEnergy(H)
        return eigenenergies

    def spectrumManifold(self,flux_points,flux_manifold,H_LC=0,H_J=null,excitation=[1]):
        """
            flux_points : inductor identifier for external introduction
            flux_manifold : [(fluxes)]
        """
        #manifold of flux space M
        Ex,E0 = full((len(excitation),len(flux_manifold)),float('nan')),[]
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        for index,fluxes in enumerate(flux_manifold):
            #import pdb;pdb.set_trace()
            external_fluxes = dict(zip(flux_points,fluxes))
            #H_J = self.josephsonEnergy(external_fluxes)
            H = H_LC + H_J(external_fluxes)
            eigenenergies = hamiltonianEnergy(H)
            
            E0.append(eigenenergies[0])
            Ex[:,index] = eigenenergies[excitation]-eigenenergies[0]
        return array(E0),array(Ex)

if __name__=='__main__':
    circuit = shuntedQubit([2,2,2])
    utils.plotMatPlotGraph(circuit.G,'circuit')
    utils.plotMatPlotGraph(circuit.spanning_tree,'spanning_tree')
    flux_manifold = zip(arange(0,1,.01))
    from test_flux_spectrum import josephsonE
    Ej0 = josephsonE(1,.125)
    Ej1 = circuit.josephsonEnergy({'I':.125})
    e1 = hamiltonianEnergy(Ej1)
    Ej2 = circuit.josephsonEnergy({'I':.0003})
    e2 = hamiltonianEnergy(Ej2)
    import ipdb; ipdb.set_trace()
