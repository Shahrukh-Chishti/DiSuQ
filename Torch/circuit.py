import networkx,copy,torch
import matplotlib.pyplot as plt

import DiSuQ.Torch.dense as Dense
import DiSuQ.Torch.sparse as Sparse
from torch import exp,det,tensor,arange,zeros,sqrt,diagonal,argsort,set_num_threads,full
from torch.linalg import eigvalsh as eigsolve,inv
from DiSuQ.Torch.components import diagonalisation,null,J,L,C,im,pi,complex

from numpy.linalg import matrix_rank,eigvalsh
from numpy import prod,array


def inverse(A):
    if det(A) == 0:
        return zeros_like(A)
    return inv(A)

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def hamiltonianEnergy(H,sort=True):
    eigenenergies = eigsolve(H)
    eigenenergies = eigenenergies.real
    if sort:
        eigenenergies = eigenenergies.sort()[0]
    return eigenenergies

def wavefunction(H,level=[0]):
    eig,vec = eigsolve(H)
    indices = argsort(eig)
    states = vec.T[indices[level]]
    return states

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

    def __init__(self,network,basis,sparse=True):
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
        self.R = self.modeTransformation().real
        self.Lo_,self.C_ = self.transformComponents()

        self.basis = basis
        # basis : list of basis_size of ith mode
        # basis : dict of O,I,J : list of basis_size
        self.sparse = sparse
        if sparse:
            self.backend = Sparse
        else:
            self.backend = Dense
            
    def initialization(self,parameters):
        for component in self.network:
            if component.__class__ == C :
                component.initCap(parameters[component.ID])
            elif component.__class__ == L :
                component.initInd(parameters[component.ID])
            elif component.__class__ == J :
                component.initJunc(parameters[component.ID])
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.Lo_,self.C_ = self.transformComponents()

    def parseCircuit(self):
        G = networkx.MultiGraph()
        for component in self.network:
            weight = 1e-3
            if component.__class__ == J:
                weight = component.energy().item() # scalar value
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

    def circuitComponents(self):
        circuit_components = dict()
        for component in self.network:
            if component.__class__ == C :
                circuit_components[component.ID] = component.capacitance().item()
            elif component.__class__ == L :
                circuit_components[component.ID] = component.inductance().item()
            elif component.__class__ == J :
                circuit_components[component.ID] = component.energy().item()
        return circuit_components

    def componentMatrix(self):
        Cn = self.nodeCapacitance()
        Cn_ = inverse(Cn)
        Rbn = self.connectionPolarity()
        Lb = self.branchInductance()
        M = self.mutualInductance()
        L_inv = inverse(Lb+M)
        Ln_ = Rbn.conj().T @ L_inv @ Rbn

        return Cn_,Ln_

    def loopFlux(self,u,v,key,external_fluxes):
        """
            external_fluxes : {identifier:flux_value}
        """
        flux = tensor(0.0)
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
                Ej.append(component.energy())
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
                    capacitance = component.energy()
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
                Lb[index,index] = component.energy()
        return Lb

    def mutualInductance(self):
        M = zeros((self.Nb,self.Nb))
        return M

    def connectionPolarity(self):
        Rbn = zeros((self.Nb,self.Nn))
        for index,(u,v,key) in self.edges_inductive.items():
            if not u==0:
                Rbn[index][self.nodes_[u]] = 1
            if not v==0:
                Rbn[index][self.nodes_[v]] = -1

        return Rbn

    def modeBasisSize(self,basis):
        n_baseO = prod(array(basis['O']))
        n_baseI = prod(array(basis['I']))
        n_baseJ = prod(array(basis['J']))

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

    def kermanBasisSize(self):
        N = 1
        basis = self.basis
        N *= prod([size for size in basis['O']])
        N *= prod([2*size+1 for size in basis['I']])
        N *= prod([2*size+1 for size in basis['J']])
        return int(N)

    def canonicalBasisSize(self):
        N = prod([2*size+1 for size in self.basis])
        return N

    def modeDistribution(self):
        Ln_ = self.Ln_
        Ni = self.islandModes()
        No = matrix_rank(Ln_.detach().numpy())
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def modeTransformation(self):
        Ln_ = self.Ln_
        R = diagonalisation(Ln_.detach(),True)
        return R

    def transformComponents(self):
        Cn_,Ln_ = self.componentMatrix()

        R = self.R
        No,Ni,Nj = self.No,self.Ni,self.Nj
        N = self.Nn
        L_ = R.conj().T @ Ln_ @ R

        Lo_ = L_[:No,:No]
        C_ = R @ Cn_ @ R.conj().T
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
        # re-calculation with parameter iteration
        DO_plus = [self.backend.displacementOscillator(basis_max,z,o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DO_minus = [self.backend.displacementOscillator(basis_max,z,-o) for o,z,basis_max in zip(O,Z,basis['O'])]
        DI_plus = [self.backend.displacementCharge(basis_max,i) for i,basis_max in zip(I,basis['I'])]
        DI_minus = [self.backend.displacementCharge(basis_max,-i) for i,basis_max in zip(I,basis['I'])]
        DJ_plus = [self.backend.displacementCharge(basis_max,j) for j,basis_max in zip(J,basis['J'])]
        DJ_minus = [self.backend.displacementCharge(basis_max,-j) for j,basis_max in zip(J,basis['J'])]
        
        Dplus = DO_plus+DI_plus+DJ_plus
        Dminus = DO_minus+DI_minus+DJ_minus
        assert len(combination)==len(Dplus)
        assert len(combination)==len(Dminus)
        return Dplus,Dminus

    def kermanHamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        N = self.kermanBasisSize()
        H = self.backend.null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                combination = self.linearCombination(i)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = self.backend.basisProduct(Dplus)
                Jminus = self.backend.basisProduct(Dminus)
            else:
                combination = self.linearCombination(i) - self.linearCombination(j)
                Dplus,Dminus = self.displacementCombination(combination)

                Jplus = self.backend.basisProduct(Dplus)
                Jminus = self.backend.basisProduct(Dminus)
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H = H-Hj

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
        Qo = [self.backend.basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [self.backend.basisQq(basis_max) for basis_max in basis['I']]
        Qj = [self.backend.basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        Co = self.backend.modeMatrixProduct(Q,Co_,Q,(0,0))

        Fo = [self.backend.basisFo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Fi = [self.backend.basisFq(basis_max) for basis_max in basis['I']]
        Fj = [self.backend.basisFq(basis_max) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        Lo = self.backend.modeMatrixProduct(F,Lo_,F,(0,0))

        Ho = (Co + Lo)/2

        Coi = self.backend.modeMatrixProduct(Q,Coi_,Q,(0,No))

        Coj = self.backend.modeMatrixProduct(Q,Coj_,Q,(0,No+Ni))

        Cij = self.backend.modeMatrixProduct(Q,Cij_,Q,(No,No+Ni))

        Hint = Coi + Coj + Cij

        Ci = self.backend.modeMatrixProduct(Q,Ci_,Q,(No,No))
        Hi = Ci/2

        Cj = self.backend.modeMatrixProduct(Q,Cj_,Q,(No+Ni,No+Ni))
        Hj = Cj/2

        return Ho+Hint+Hi+Hj
    
    def kermanChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn)
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        charge = self.R@charge
        
        No,Ni,Nj = self.No,self.Ni,self.Nj
        
        Qo = [self.backend.basisQq(basis_max) for basis_max in basis['O']]
        Qi = [self.backend.basisQq(basis_max) for basis_max in basis['I']]
        Qj = [self.backend.basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj
        Io = [self.backend.identity(2*basis_max+1)*0.0 for basis_max in basis['O']]
        Ii = [self.backend.identity(2*basis_max+1)*charge[index+No] for index,basis_max in enumerate(basis['I'])]
        Ij = [self.backend.identity(2*basis_max+1)*charge[index+No+Ni] for index,basis_max in enumerate(basis['J'])]
        I = Io + Ii + Ij
        
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = self.C_
        
        Coi = self.backend.modeMatrixProduct(Q,Coi_,I,(0,No))
        Coj = self.backend.modeMatrixProduct(Q,Coj_,I,(0,No+Ni))
        
        Cij = self.backend.modeMatrixProduct(I,Coj_,Q,(No+Ni,No))
        Cij += self.backend.modeMatrixProduct(Q,Coj_,I,(No+Ni,No))
        Cij -= self.backend.modeMatrixProduct(I,Coj_,I,(No+Ni,No))
        
        Ci = self.backend.modeMatrixProduct(I,Ci_,Q,(No,No))
        Ci += self.backend.modeMatrixProduct(Q,Ci_,I,(No,No))
        Ci += self.backend.modeMatrixProduct(I,Ci_,I,(No,No))
        
        Cj = self.backend.modeMatrixProduct(I,Cj_,Q,(No+Ni,No+Ni))
        Cj += self.backend.modeMatrixProduct(Q,Cj_,I,(No+Ni,No+Ni))
        Cj += self.backend.modeMatrixProduct(I,Cj_,I,(No+Ni,No+Ni))

        H = Coi + Coj + Ci/2 + Cj/2
        return -H

    def josephsonFlux(self,external_fluxes=dict()):
        basis = self.basis
        Z = self.modeImpedance() * 2 # cooper pair factor
        Dplus = [self.backend.displacementFlux(basis_max,1) for z,basis_max in zip(Z,basis)]
        Dminus = [self.backend.displacementFlux(basis_max,-1) for z,basis_max in zip(Z,basis)]
        edges,Ej = self.josephsonComponents()
        N = self.canonicalBasisSize()
        H = self.backend.null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus = self.backend.basisProduct(Dplus,[i])
                Jminus = self.backend.basisProduct(Dminus,[i])
            else:
                Jplus = self.backend.crossBasisProduct(Dplus,Dminus,i,j)
                Jminus = self.backend.crossBasisProduct(Dplus,Dminus,j,i)
                #assert (Jminus == Jplus.conj().T).all()
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H = H - Hj

        return H/2

    def josephsonOscillator(self,external_fluxes=dict()):
        basis = self.basis
        Z = self.modeImpedance() * 2 # cooper pair factor
        Dplus = [self.backend.displacementOscillator(basis_max,z,1) for z,basis_max in zip(Z,basis)]
        Dminus = [self.backend.displacementOscillator(basis_max,z,-1) for z,basis_max in zip(Z,basis)]
        edges,Ej = self.josephsonComponents()
        N = prod(basis)
        H = self.backend.null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus = self.backend.basisProduct(Dplus,[i])
                Jminus = self.backend.basisProduct(Dminus,[i])
            else:
                Jplus = self.backend.crossBasisProduct(Dplus,Dminus,i,j)
                Jminus = self.backend.crossBasisProduct(Dplus,Dminus,j,i)
                #assert (Jminus == Jplus.conj().T).all()
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H = H - Hj

        return H/2

    def josephsonCharge(self,external_fluxes=dict()):
        basis = self.basis
        Dplus = [self.backend.chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [self.backend.chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        N = self.canonicalBasisSize()
        H = self.backend.null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus = self.backend.basisProduct(Dplus,[i])
                Jminus = self.backend.basisProduct(Dminus,[i])
            else:
                Jplus = self.backend.crossBasisProduct(Dplus,Dminus,i,j)
                Jminus = self.backend.crossBasisProduct(Dplus,Dminus,j,i)
                #assert (Jminus == Jplus.conj().T).all()
                
            #print(E,flux)
            Hj = E*(Jplus*phase(flux) + Jminus*phase(-flux))
            H = H-Hj

        return H/2

    def fluxInducerEnergy(self):
        basis = self.basis
        fluxModes = [basisPj(basis_max) for basis_max in basis]
        edges,L_ext = self.fluxBiasComponents()
        H = tensor([[0.0]])
        for (u,v,key),L in zip(edges,L_ext):
            i,j = self.nodes_[u],self.nodes_[v]
            #print(u,v,i,j,L)
            if i<0 or j<0 :
                # grounded inducer
                i = max(i,j)
                P = self.backend.basisProduct(fluxModes,[i])
            else:
                P = self.backend.basisProduct(fluxModes,[i]) - self.backend.basisProduct(fluxModes,[j])
            H = H + P@P / 2 / L
        return H

    def oscillatorHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        Q = [self.backend.basisQo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        F = [self.backend.basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]

        H_C = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H

    def fluxHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [self.backend.basisQf(basis_max) for basis_max in basis]
        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H_C = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H

    def chargeHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [self.backend.basisQq(basis_max) for basis_max in basis]
        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H_C = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)

        H = (H_C+H_L)/2

        return H
    
    def chargeChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn)
        basis = self.basis
        Cn_ = self.Cn_
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
            
        Q = [self.backend.basisQq(basis_max) for basis_max in basis]
        I = [self.backend.identity(2*basis_max+1,complex)*charge[index] for index,basis_max in enumerate(basis)]
        #import pdb;pdb.set_trace()
        H = self.backend.modeMatrixProduct(Q,Cn_,I)
        H += self.backend.modeMatrixProduct(I,Cn_,Q)
        H -= self.backend.modeMatrixProduct(I,Cn_,I)
        
        return -H/2.

    def potentialOscillator(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        F = [self.backend.basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonOscillator(external_fluxes)
        return H_L+H_J

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonCharge(external_fluxes)
        return H_L+H_J

    def potentialFlux(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H_L = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H_J = self.josephsonFlux(external_fluxes)
        return H_L+H_J

    def circuitEnergy(self,H_LC=tensor(0.0),H_J=None,external_fluxes=dict(),grad=True):
        if H_J is None:
            H_J = null(H_LC)
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        H = H_LC + H_J(external_fluxes)
        if grad:
            if self.sparse:
                H = H.to_dense()
            eigenenergies = hamiltonianEnergy(H)
        else:
            if self.sparse:
                H = self.backend.scipyfy(H)
                eigenenergies = self.backend.sparse.linalg.eigsh(H,return_eigenvectors=False)
            else:
                H = H.detach().numpy()
                eigenenergies = eigvalsh(H)
        
        return eigenenergies

    def spectrumManifold(self,flux_points,flux_manifold,H_LC=tensor(0.0),H_J=None,excitation=[1]):
        """
            flux_points : inductor identifier for external introduction
            flux_manifold : [(fluxes)]
        """
        if H_J is None:
            H_J = null(H_LC)
        #manifold of flux space M
        Ex,E0 = full((len(excitation),len(flux_manifold)),float('nan')),[]
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        for index,fluxes in enumerate(flux_manifold):
            external_fluxes = dict(zip(flux_points,fluxes))
            #H_J = self.josephsonEnergy(external_fluxes)
            H = H_LC + H_J(external_fluxes)
            eigenenergies = hamiltonianEnergy(H)
            
            E0.append(eigenenergies[0])
            Ex[:,index] = eigenenergies[excitation]-eigenenergies[0]
        return E0,Ex
    
    def fluxScape(self,flux_points,flux_manifold):
        H_LC = self.kermanHamiltonianLC()
        H_J = self.kermanHamiltonianJosephson
        E0,EI = self.spectrumManifold(flux_point,flux_profile,H_LC,H_J,excitation=1)
        E0,EII = self.spectrumManifold(flux_point,flux_profile,H_LC,H_J,excitation=2)
        EI = tensor(EI).detach().numpy()
        EII = tensor(EII).detach().numpy()
        return EI,EII

if __name__=='__main__':
    import models
    circuit = models.shuntedQubit([2,2,2],sparse=False)
    flux_manifold = zip(arange(0,1,.01))
    H_LC = circuit.chargeHamiltonianLC()
    H_J = circuit.josephsonCharge({'I':tensor(.125)})
    circuit.basis = {'O':[2],'I':[],'J':[2,2]}
    H_J = circuit.kermanHamiltonianJosephson({'I':tensor(.225)})
    import ipdb; ipdb.set_trace()
    H_LC = circuit.kermanHamiltonianLC()
    e1 = hamiltonianEnergy(H_J)
    Ej2 = circuit.josephsonCharge({'I':tensor(.0003)})
    e2 = hamiltonianEnergy(Ej2)
