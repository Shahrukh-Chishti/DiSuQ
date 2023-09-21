import networkx,copy,torch
import matplotlib.pyplot as plt

import DiSuQ.Torch.dense as Dense
import DiSuQ.Torch.sparse as Sparse
from torch import exp,det,tensor,arange,zeros,zeros_like,sqrt,diagonal,argsort,lobpcg,set_num_threads,full as full_torch
from torch.linalg import eigvalsh as eigsolve,inv,eigh
from DiSuQ.Torch.components import diagonalisation,null,J,L,C,im,pi,complex
from time import perf_counter
from numpy.linalg import matrix_rank
from numpy.linalg import eigvalsh
from numpy import prod,flip,array,sort,full as full_numpy


def inverse(A,zero=1e-15):
    if det(A) == 0:
        #return zeros_like(A)
        D = A.diag()
        A[D==0,D==0] = tensor(1/zero)
        import pdb;pdb.set_trace()
    try:
        A = inv(A)
    except:
        import pdb;pdb.set_trace()
    #import pdb;pdb.set_trace()    
    #A[A<=zero] = tensor(0.)
    return A

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def hamiltonianEnergy(H,sort=True):
    eigenenergies = eigsolve(H)
    return eigenenergies

def wavefunction(H,level=[0]):
    eig,vec = eigh(H)
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

    def __init__(self,network,basis,sparse=True,pairs=dict()):
        self.network = network
        self.G = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges,self.edges_inductive = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nb = len(self.edges_inductive)
        self.pairs = pairs
        self.symmetrize(self.pairs)
        self.Cn_,self.Ln_ = self.componentMatrix()
        # default : Kerman transformation
        #self.No,self.Ni,self.Nj = self.kermanDistribution()
        self.R = self.kermanTransform().real
        self.L_,self.C_ = modeTransformation()
        #self.Lo_,self.C_ = self.kermanComponents()

        self.basis = basis
        # basis : list of basis_size of ith mode
        # basis : dict of O,I,J : list of basis_size
        self.sparse = sparse
        if sparse:
            self.backend = Sparse
        else:
            self.backend = Dense
            
    def initialization(self,parameters):
        # parameters : GHz unit
        for component in self.network:
            if component.__class__ == C :
                component.initCap(parameters[component.ID])
            elif component.__class__ == L :
                component.initInd(parameters[component.ID])
            elif component.__class__ == J :
                component.initJunc(parameters[component.ID])
                
        self.symmetrize(self.pairs)
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = modeTransformation()
        #self.Lo_,self.C_ = self.transformComponents()
        
    def symmetrize(self,pairs):
        components = self.circuitComposition()
        for slave,master in pairs.items():
            master = components[master]
            slave = components[slave]
            if slave.__class__ == C :
                slave.cap = master.cap
            elif slave.__class__ == L :
                slave.ind = master.ind
            elif slave.__class__ == J :
                slave.jo = master.jo

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
    
    def circuitState(self):
        parameters = {}
        for component in self.network:
            parameters[component.ID] = component.energy().item()
        return parameters
    
    def circuitComposition(self):
        components = dict()
        for component in self.network:
            components[component.ID] = component
        return components

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
        assert not det(Cn)==0
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
                    capacitance = component.capacitance()
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
                Lb[index,index] = component.inductance()
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
    
    def canonicalBasisSize(self):
        N = prod([2*size+1 for size in self.basis])
        return N

    def kermanBasisSize(self):
        N = 1
        basis = self.basis
        N *= prod([size for size in basis['O']])
        N *= prod([2*size+1 for size in basis['I']])
        N *= prod([2*size+1 for size in basis['J']])
        return int(N)

    def kermanDistribution(self):
        Ln_ = self.Ln_
        Ni = self.islandModes()
        No = matrix_rank(Ln_.detach().numpy())
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def kermanTransform(self):
        Ln_ = self.Ln_
        R = diagonalisation(Ln_.detach(),True)
        return R
    
    def kermanComponents(self):
        L_,C_ = self.L_,self.C_
        No,Ni,Nj = self.kermanDistribution() #self.No,self.Ni,self.Nj
        N = self.Nn

        Lo_ = L_[:No,:No]
        
        Co_ = C_[:No,:No]
        Coi_ = C_[:No,No:No+Ni]
        Coj_ = C_[:No,No+Ni:]
        Ci_ = C_[No:No+Ni,No:No+Ni]
        Cij_ = C_[No:No+Ni,No+Ni:]
        Cj_ = C_[No+Ni:,No+Ni:]

        C_ = Co_,Coi_,Coj_,Ci_,Cij_,Cj_

        return Lo_,C_
    
    def modeTransformation(self):
        Cn_,Ln_ = self.componentMatrix()
        R = self.R
        L_ = inv(R.T) @ Ln_ @ inv(R)
        C_ = R @ Cn_ @ R.T
        return L_,C_    

    def modeImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        return impedance

    def oscillatorImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis['O']))]
        return impedance

    def linearCombination(self,index):
        invR = inv(self.R)
        combination = invR[index]
        assert len(combination) == self.Nn
        return combination

    def displacementCombination(self,combination):
        basis = self.basis
        No,Ni,Nj = self.kermanDistribution()#self.No,self.Ni,self.Nj
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
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

        return H/2

    def kermanHamiltonianLC(self):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        basis = self.basis
        Lo_,C_ = self.kermanComponents()

        #Lo_ = self.Lo_
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        n_baseO,n_baseI,n_baseJ = self.modeBasisSize(basis)
        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj

        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Qo = [self.backend.basisQo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Qi = [self.backend.basisQq(basis_max) for basis_max in basis['I']]
        Qj = [self.backend.basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        H = self.backend.modeMatrixProduct(Q,Co_,Q,(0,0))/2

        Fo = [self.backend.basisFo(basis_max,Zi) for Zi,basis_max in zip(Z,basis['O'])]
        Fi = [self.backend.basisFq(basis_max) for basis_max in basis['I']]
        Fj = [self.backend.basisFq(basis_max) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        H += self.backend.modeMatrixProduct(F,Lo_,F,(0,0))/2

        H += self.backend.modeMatrixProduct(Q,Coi_,Q,(0,No))
        H += self.backend.modeMatrixProduct(Q,Coj_,Q,(0,No+Ni))
        H += self.backend.modeMatrixProduct(Q,Cij_,Q,(No,No+Ni))

        H += self.backend.modeMatrixProduct(Q,Ci_,Q,(No,No))/2
        H += self.backend.modeMatrixProduct(Q,Cj_,Q,(No+Ni,No+Ni))/2

        return H
    
    def kermanChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn)
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        charge = self.R@charge
        
        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj
        
        Qo = [self.backend.basisQq(basis_max) for basis_max in basis['O']]
        Qi = [self.backend.basisQq(basis_max) for basis_max in basis['I']]
        Qj = [self.backend.basisQq(basis_max) for basis_max in basis['J']]
        Q = Qo + Qi + Qj
        Io = [self.backend.identity(2*basis_max+1)*0.0 for basis_max in basis['O']]
        Ii = [self.backend.identity(2*basis_max+1)*charge[index+No]*2 for index,basis_max in enumerate(basis['I'])]
        Ij = [self.backend.identity(2*basis_max+1)*charge[index+No+Ni]*2 for index,basis_max in enumerate(basis['J'])]
        I = Io + Ii + Ij
        
        Lo_,C_ = self.kermanComponents()
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        
        H = self.backend.modeMatrixProduct(Q,Coi_,I,(0,No))
        H += self.backend.modeMatrixProduct(Q,Coj_,I,(0,No+Ni))
        
        H += self.backend.modeMatrixProduct(I,Coj_,Q,(No+Ni,No))
        H += self.backend.modeMatrixProduct(Q,Coj_,I,(No+Ni,No))
        H -= self.backend.modeMatrixProduct(I,Coj_,I,(No+Ni,No))
        
        H += self.backend.modeMatrixProduct(I,Ci_,Q,(No,No))/2
        H += self.backend.modeMatrixProduct(Q,Ci_,I,(No,No))/2
        H += self.backend.modeMatrixProduct(I,Ci_,I,(No,No))/2
        
        H += self.backend.modeMatrixProduct(I,Cj_,Q,(No+Ni,No+Ni))/2
        H += self.backend.modeMatrixProduct(Q,Cj_,I,(No+Ni,No+Ni))/2
        H += self.backend.modeMatrixProduct(I,Cj_,I,(No+Ni,No+Ni))/2

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
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

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
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

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
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

        return H/2
    
    def josephsonMixed(self,basis,n_trunc):
        assert len(basis)==len(n_trunc)
        Dplus,Dminus = [],[]
        Z = self.modeImpedance()
        for base,n,z in zip(basis,n_trunc,Z):
            if base=='o':
                Dplus.append(self.backend.displacementOscillator(n,z,1))
                Dminus.append(self.backend.displacementOscillator(n,z,-1))
            elif base == 'q':
                Dplus.append(self.backend.chargeDisplacePlus(n))
                Dminus.append(self.backend.chargeDisplaceMinus(n))
            elif base == 'f':
                Dplus.append(self.backend.displacementFlux(n,1))
                Dminus.append(self.backend.displacementFlux(n,-1))
                
        assert len(Dplus) == len(basis)
        N = prod([len(D) for D in Dplus])
        def Hj(external_fluxes=dict()):
            edges,Ej = self.josephsonComponents()
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
                H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

            return H/2
        return Hj
        

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

        H = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H += self.backend.modeMatrixProduct(F,Ln_,F)

        return H/2

    def fluxHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis

        Q = [self.backend.basisQf(basis_max) for basis_max in basis]
        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H += self.backend.modeMatrixProduct(F,Ln_,F)

        return H/2

    def chargeHamiltonianLC(self):
        """
            basis : [basis_size] charge
        """
        Cn_,Ln_ = self.Cn_,self.Ln_
        basis = self.basis
        Q = [self.backend.basisQq(basis_max) for basis_max in basis]
        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H += self.backend.modeMatrixProduct(F,Ln_,F)

        return H/2
    
    def mixedHamiltonianLC(self,basis,n_trunc):
        assert len(basis)==len(n_trunc)
        Cn_,Ln_ = self.Cn_,self.Ln_
        Q,F = [],[]
        Z = self.modeImpedance()
        for base,n,z in zip(basis,n_trunc,Z):
            if base=='o':
                Q.append(self.backend.basisQo(n,z))
                F.append(self.backend.basisFo(n,z))
            elif base == 'q':
                Q.append(self.backend.basisQq(n))
                F.append(self.backend.basisFq(n))
            elif base == 'f':
                Q.append(self.backend.basisQf(n))
                F.append(self.backend.basisFf(n))
                
        assert len(Q) == len(basis)
        assert len(F) == len(basis)
        H = self.backend.modeMatrixProduct(Q,Cn_,Q)
        H += self.backend.modeMatrixProduct(F,Ln_,F)

        return H/2
    
    def chargeChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn)
        basis = self.basis
        Cn_ = self.Cn_
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
            
        Q = [self.backend.basisQq(basis_max) for basis_max in basis]
        I = [self.backend.identity(2*basis_max+1,complex)*charge[index]*2 for index,basis_max in enumerate(basis)]
        H = self.backend.modeMatrixProduct(Q,Cn_,I)
        H += self.backend.modeMatrixProduct(I,Cn_,Q)
        H += self.backend.modeMatrixProduct(I,Cn_,I)
        
        return H/2.

    def potentialOscillator(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        F = [self.backend.basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonOscillator(external_fluxes)
        return H

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonCharge(external_fluxes)
        return H

    def potentialFlux(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonFlux(external_fluxes)
        return H
    
    def operatorExpectation(self,bra,O,mode,ket):
        basis = self.basis
        O = [O(basis_max) for basis_max in basis]
        O = self.backend.basisProduct(O,[mode])
        return bra.conj()@ O@ ket

    def circuitEnergy(self,H_LC=tensor(0.0),H_J=None,external_fluxes=dict(),grad=True):
        ## this could be improved : removing if clause, sub-class, sparse/dense and grad/numer segregation
        if H_J is None:
            H_J = null(H_LC)
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        H = H_LC + H_J(external_fluxes)
        if grad:
            if self.sparse:
                eigenenergies = lobpcg(H.to(float),k=3,largest=False)[0]
            else:
                eigenenergies = hamiltonianEnergy(H)
        else:
            if self.sparse:
                H = self.backend.scipyfy(H)
                eigenenergies = self.backend.sparse.linalg.eigsh(H,return_eigenvectors=False,which='SA')
                eigenenergies = sort(eigenenergies)
            else:
                H = H.detach().numpy() 
                eigenenergies = eigvalsh(H)
        
        return eigenenergies

    def spectrumManifold(self,flux_points,flux_manifold,H_LC=tensor(0.0),H_J=None,excitation=[1],grad=True,log=False):
        """
            flux_points : inductor identifier for external introduction
            flux_manifold : [(fluxes)]
        """
        if H_J is None:
            H_J = null(H_LC)
        #manifold of flux space M
        if not grad:
            Ex = full_numpy((len(excitation),len(flux_manifold)),float('nan'))
        else:
            Ex = full_torch((len(excitation),len(flux_manifold)),float('nan'))
        E0 = []
        #H_LC = self.hamiltonianLC()
        #H_ext = self.fluxInducerEnergy()
        start = perf_counter()
        for index,fluxes in enumerate(flux_manifold):
            if log:
                if index%50 == 0:
                    print(index,'\t',perf_counter()-start)
            external_fluxes = dict(zip(flux_points,fluxes))
            #H_J = self.josephsonEnergy(external_fluxes)
            #H = H_LC + H_J(external_fluxes)
            eigenenergies = self.circuitEnergy(H_LC,H_J,external_fluxes,grad)
            #eigenenergies = hamiltonianEnergy(H)
            
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
