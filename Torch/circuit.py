import networkx,copy,torch
import matplotlib.pyplot as plt
from contextlib import nullcontext
import DiSuQ.Torch.dense as Dense
import DiSuQ.Torch.sparse as Sparse
from torch import exp,det,tensor,tile,arange,ones,zeros,zeros_like,sqrt,diagonal,argsort,lobpcg,set_num_threads,full as full_torch
from torch.linalg import eigvalsh,inv,eigh
from DiSuQ.Torch.components import diagonalisation,null,J,L,C,im,pi,complex,float
from DiSuQ.Torch.components import COMPILER_BACKEND
from time import perf_counter
from numpy.linalg import matrix_rank
#from numpy.linalg import eigvalsh
from numpy import prod,flip,array,sort,full as full_numpy
from torch import nn
from torch.nn.parameter import Parameter

### Computational Sub-routines

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

def impedanceUpdateCap(Co_,Z):
    N = len(Z)
    Z = 1/sqrt(Z)
    Co_ = Z.view(N,1).expand(N,N)*Co_*Z.expand(N,N)
    return Co_

def impedanceUpdateInd(Lo_,Z):
    N = len(Z)
    Z = sqrt(Z)
    Lo_ = Z.view(N,1).expand(N,N)*Lo_*Z.expand(N,N)
    return Lo_

def coeffProduct(H,Coeff,M):
    nA,nB = Coeff.shape
    for i in range(nA):
        for j in range(nB):
            #if not M[i,j] == 0:
            H += Coeff[i,j]*M[(i,j)]
    return H

### Generic Diagonalization

def hamiltonianEnergy(H):
    eigenenergies = eigsolve(H)
    return eigenenergies

def wavefunction(H,level=[0]):
    eig,vec = eigh(H)
    states = vec.T[level]
    return states

class Circuit(nn.Module):
    """
        * no external fluxes must be in parallel : redundant
        * no LC component must be in parallel : redundant
        * only Josephson elements allowed in parallel
    """
    """
        * u,v : indices of raw graph : GL, minimum_spanning_tree
        * i,j : indices of indexed graph : mode correspondence
    """

    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__()
        # circuit network
        self.network = network
        self.control_iD = control_iD
        self.G = self.parseCircuit()
        self.spanning_tree = self.spanningTree()
        self.nodes,self.nodes_ = self.nodeIndex()
        self.edges,self.edges_inductive = self.edgesIndex()
        self.Nn = len(self.nodes)
        self.Ne = len(self.edges)
        self.Nb = len(self.edges_inductive)
        self.pairs = pairs
        self.symmetrize(self.pairs)
        # circuit components
        self.Cn_,self.Ln_ = self.componentMatrix()

        self.basis = basis
        # basis : list of basis_size of ith mode
        # basis : dict of O,I,J : list of basis_size
        self.sparse = sparse
        if sparse:
            self.backend = Sparse
        else:
            self.backend = Dense
        self.device = device
        self.null_flux = tensor(0.,device=self.device)
        self.null = self.backend.null(self.basisSize(),device=self.device)
        
        self.spectrum_limit = 4
        self.vectors_calc = False
        self.grad_calc = True

        self.forward = self.spectrumManifold

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

    # check if DDP module latest version build parameters

    def named_parameters(self,subspace=(),recurse=False):
        parameters = []; IDs = []
        slaves = self.pairs.keys()
        for component in self.network:
            if component.ID in subspace or len(subspace)==0:
                if component.__class__ == C :
                    parameter = component.cap
                elif component.__class__ == L :
                    parameter = component.ind
                elif component.__class__ == J :
                    parameter = component.jo
                if not component.ID in slaves:
                    IDs.append(component.ID)
                    parameters.append(parameter)
        return zip(IDs,parameters)

    def parameters(self):
        independent = []
        slaves = self.pairs.keys()
        for component in self.network:
            if component.__class__ == C :
                parameter = component.cap
            elif component.__class__ == L :
                parameter = component.ind
            elif component.__class__ == J :
                parameter = component.jo
            if not component.ID in slaves:
                independent.append(parameter)
        return independent
        
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
        flux = self.null_flux.clone()
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
        Cn = zeros((self.Nn,self.Nn)) ##
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
        Lb = zeros((self.Nb,self.Nb)) ##
        #fill_diagonal(Lb,L_limit)
        for index,(u,v,key) in self.edges_inductive.items():
            component = self.G[u][v][key]['component']
            if component.__class__ == L:
                #if not component.external:
                Lb[index,index] = component.inductance()
        return Lb

    def mutualInductance(self):
        M = zeros((self.Nb,self.Nb)) ##
        return M

    def connectionPolarity(self):
        Rbn = zeros((self.Nb,self.Nn))
        for index,(u,v,key) in self.edges_inductive.items():
            if not u==0:
                Rbn[index][self.nodes_[u]] = 1
            if not v==0:
                Rbn[index][self.nodes_[v]] = -1

        return Rbn

    def islandModes(self):
        islands = self.graphGL(elements=[C])
        islands = networkx.connected_components(islands)
        islands = list(islands)
        Ni = 0
        for sub in islands:
            if 0 not in sub:
                Ni += 1
        return Ni

    def fluxInducerEnergy(self):
        basis = self.basis
        fluxModes = [basisPj(basis_max) for basis_max in basis]
        edges,L_ext = self.fluxBiasComponents()
        H = self.backend.null() #tensor([[0.0]])
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
    
    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def circuitHamiltonian(self,external_flux):
        # torch reconstruct computation graph repeatedly
        H = self.hamiltonianLC()
        H += self.hamiltonianJosephson(external_flux)
        return H
    
    def controlData(self,control):
        if len(self.control_iD) == 0:
            return dict()
        assert len(control) == len(self.control_iD)
        control = dict(zip(self.control_iD,control))
        return control
    
    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def eigenSpectrum(self,control):
        control = self.controlData(control)
        with torch.inference_mode() if self.grad_calc is False else nullcontext() as null:
            H = self.circuitHamiltonian(control)
            if self.vectors_calc:
                if H.dtype is float:
                    # w/o : flux controls or Fourier conjugate
                    spectrum,states = lobpcg(H,k=self.spectrum_limit,largest=False,method='ortho')
                else:
                    spectrum,states = eigh(H)
                    spectrum = spectrum[:self.spectrum_limit]
                    states = states[:self.spectrum_limit]
            else:
                # stable eigenvalues over degeneracy limits
                spectrum = eigvalsh(H); states = None
                spectrum = spectrum[:self.spectrum_limit]
        return spectrum,states
    
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
                # illegal type casting
                eigenenergies = lobpcg(H.to(float),k=4,largest=False)[0]
            else:
                eigenenergies = eigsolve(H)
        else:
            if self.sparse:
                H = self.backend.scipyfy(H)
                eigenenergies = self.backend.sparse.linalg.eigsh(H,return_eigenvectors=False,which='SA')
                eigenenergies = sort(eigenenergies)
            else:
                H = H.detach().numpy()
                eigenenergies = eigvalsh(H)
        return eigenenergies

    def spectrumManifold(self,manifold):
        # sequential execution within batch of control points
        Spectrum = [] # check autograd over creation
        for control in manifold:
            spectrum,states = self.eigenSpectrum(control)
            Spectrum.append((spectrum,states))
        return Spectrum

class Kerman(Circuit):
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)
        self.No,self.Ni,self.Nj = self.kermanDistribution()
        self.R = self.kermanTransform().real
        self.L_,self.C_ = self.modeTransformation()
        #self.Lo_,self.C_ = self.kermanComponents()
        # JIT-compilation indifferent pers-initialization
        self.Q,self.F,self.DI_plus,self.DI_minus,self.DJ_plus,self.DJ_minus = self.operatorInitialization()
        self.Qoo,self.Foo,self.Qoi,self.Qoj,self.Qij,self.Qij,self.Qii,self.Qjj = self.oscillatorInitialization()
        self.Dplus,self.Dminus = self.josephsonInitialization()

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
        # recalculate capacitances and inductances
        # for 1) nodes-formalism 2) mode-formalism
        # Transformation Invariant
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        #self.Lo_,self.C_ = self.transformComponents()

    def basisSize(self,modes=False):
        N = dict()
        basis = self.basis
        N['O'] = [size for size in basis['O']]
        N['I'] = [2*size+1 for size in basis['I']]
        N['J'] = [2*size+1 for size in basis['J']]
        if modes:
            return N
        N = prod(N['O'])*prod(N['I'])*prod(N['J'])
        return int(N)

    def kermanDistribution(self):
        Ln_ = self.Ln_
        Ni = self.islandModes()
        No = matrix_rank(Ln_.detach().numpy())
        Nj = self.Nn - Ni - No
        return No,Ni,Nj

    def kermanTransform(self):
        Ln_ = self.Ln_
        R = diagonalisation(Ln_.detach(),True).to(float)
        return R
    
    def kermanComponents(self):
        L_,C_ = self.L_,self.C_
        No,Ni,Nj = self.No,self.Ni,self.Nj #self.kermanDistribution()
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
        Cn_,Ln_ = self.Cn_,self.Ln_ #componentMatrix()
        R = self.R
        L_ = inv(R.T) @ Ln_ @ inv(R)
        C_ = R @ Cn_ @ R.T
        return L_,C_

    def oscillatorImpedance(self):
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()
        impedance = [sqrt(C_[0][i,i]/Lo_[i,i]) for i in range(len(self.basis['O']))]
        return impedance

    def linearCombination(self,edge):
        invR = inv(self.R)
        i,j = edge
        if i<0 or j<0 :
            # grounded josephson
            i = max(i,j)
            combination = invR[i]
        else:
            combination = invR[i] - invR[j]
        return combination

    def operatorInitialization(self):
        basis = self.basis
        No,Ni,Nj = self.No,self.Ni,self.Nj

        Qo = [self.backend.basisQo(basis_max,tensor(1.),self.device) for basis_max in basis['O']]
        Qi = [self.backend.basisQq(basis_max,self.device) for basis_max in basis['I']]
        Qj = [self.backend.basisQq(basis_max,self.device) for basis_max in basis['J']]
        Q = Qo + Qi + Qj

        Fo = [self.backend.basisFo(basis_max,tensor(1.),self.device) for basis_max in basis['O']]
        Fi = [self.backend.basisFq(basis_max,self.device) for basis_max in basis['I']]
        Fj = [self.backend.basisFq(basis_max,self.device) for basis_max in basis['J']]
        F = Fo + Fi + Fj

        DI_plus,DI_minus = dict(),dict()
        DJ_plus,DJ_minus = dict(),dict()
        edges,Ej = self.josephsonComponents()
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            edge = u,v
            combination = self.linearCombination((i,j))
            I = combination[No:No+Ni]
            J = combination[No+Ni:]

            DI_plus[edge] = [self.backend.displacementCharge(basis_max,i,self.device) for i,basis_max in zip(I,basis['I'])]
            DI_minus[edge] = [self.backend.displacementCharge(basis_max,-i,self.device) for i,basis_max in zip(I,basis['I'])]
            DJ_plus[edge] = [self.backend.displacementCharge(basis_max,j,self.device) for j,basis_max in zip(J,basis['J'])]
            DJ_minus[edge] = [self.backend.displacementCharge(basis_max,-j,self.device) for j,basis_max in zip(J,basis['J'])]

        return Q,F,DI_plus,DI_minus,DJ_plus,DJ_minus

    def oscillatorInitialization(self):
        # persistence/initialization of mode matrix product
        No,Ni,Nj = self.No,self.Ni,self.Nj
        Q,F = self.Q,self.F
        Qoo,Foo = dict(),dict()
        Qoi,Qoj,Qij = dict(),dict(),dict()
        Qii,Qjj = dict(),dict()
        for i in range(No):
            for j in range(No):
                Qoo[(i,j)] = self.backend.modeProduct(Q,i,Q,j)
                Foo[(i,j)] = self.backend.modeProduct(F,i,F,j)
            for j in range(Ni):
                Qoi[(i,j)] = self.backend.modeProduct(Q,i,Q,j+No)
            for j in range(Nj):
                Qoj[(i,j)] = self.backend.modeProduct(Q,i,Q,j+No+Ni)

        for i in range(Ni):
            for j in range(Ni):
                Qii[(i,j)] = self.backend.modeProduct(Q,i+No,Q,j+No)
            for j in range(Nj):
                Qii[(i,j)] = self.backend.modeProduct(Q,i+No,Q,j+No+Ni)

        for i in range(Nj):
            for j in range(Nj):
                Qjj[(i,j)] = self.backend.modeProduct(Q,i+No+Ni,Q,j+No+Ni)

        return Qoo,Foo,Qoi,Qoj,Qij,Qij,Qii,Qjj

    def josephsonInitialization(self):
        edges,Ej = self.josephsonComponents()
        Dplus,Dminus = dict(),dict()
        for (u,v,key),E in zip(edges,Ej):
            edge = u,v
            Dplus[edge] = self.backend.basisProduct(self.DI_plus[edge]+self.DJ_plus[edge])
            Dminus[edge] = self.backend.basisProduct(self.DI_minus[edge]+self.DJ_minus[edge])
        return Dplus,Dminus

    def displacementCombination(self,combination,edge):
        basis = self.basis
        No,Ni,Nj = self.No,self.Ni,self.Nj
        O = combination[:No]
        Z = self.oscillatorImpedance() * 2 # cooper pair factor
        # oscillator-calculation
        DO_plus = [self.backend.displacementOscillator(basis_max,z,o,self.device) for o,z,basis_max in zip(O,Z,basis['O'])]
        DO_minus = [self.backend.displacementOscillator(basis_max,z,-o,self.device) for o,z,basis_max in zip(O,Z,basis['O'])]

        Dplus = DO_plus+[self.Dplus[edge]]
        Dminus = DO_minus+[self.Dminus[edge]]

        return Dplus,Dminus

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        N = self.basisSize()
        H = self.backend.null(N)
        for (u,v,key),E in zip(edges,Ej):
            i,j = self.nodes_[u],self.nodes_[v]
            flux = self.loopFlux(u,v,key,external_fluxes)
            combination = self.linearCombination((i,j))
            Dplus,Dminus = self.displacementCombination(combination,(u,v))

            Jplus = self.backend.basisProduct(Dplus)
            Jminus = self.backend.basisProduct(Dminus)
            H -= E*(Jplus*phase(flux) + Jminus*phase(-flux))

        return H/2

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianLC(self):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()

        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
        No,Ni,Nj = self.No,self.Ni,self.Nj

        # impedance update to Cap/Ind matrix - Oscillator mode
        Z = sqrt(diagonal(Co_)/diagonal(Lo_))
        Co_ = impedanceUpdateCap(Co_,Z)
        Lo_ = impedanceUpdateInd(Lo_,Z)

        H = self.backend.null(self.basisSize())

        H = coeffProduct(H,Co_,self.Qoo)/2

        H = coeffProduct(H,Lo_,self.Foo)/2

        H = coeffProduct(H,Coi_,self.Qoi)
        H = coeffProduct(H,Coj_,self.Qoj)
        H = coeffProduct(H,Cij_,self.Qij)

        H = coeffProduct(H,Ci_,self.Qii)/2
        H = coeffProduct(H,Cj_,self.Qjj)/2

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
        
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
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

class Oscillator(Circuit):

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

    def potentialOscillator(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        F = [self.backend.basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonOscillator(external_fluxes)
        return H

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

class Flux(Circuit):

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

    def potentialFlux(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonFlux(external_fluxes)
        return H

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

class Charge(Circuit):
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)
        # JIT-compilation indifferent pers-initialization
        self.Q,self.F,self.D_plus,self.D_minus = self.operatorInitialization()
        self.QQ,self.FF = self.oscillatorInitialization()
        self.Jplus,self.Jminus = self.josephsonInitialization()

    def basisSize(self,modes=False):
        N = [2*size+1 for size in self.basis]
        if modes:
            return N
        return prod(N)

    def operatorInitialization(self):
        basis = self.basis
        Q = [self.backend.basisQq(basis_max,self.device) for basis_max in basis]
        F = [self.backend.basisFq(basis_max,self.device) for basis_max in basis]
        Dplus = [self.backend.chargeDisplacePlus(basis_max,self.device) for basis_max in basis]
        Dminus = [self.backend.chargeDisplaceMinus(basis_max,self.device) for basis_max in basis]
        return Q,F,Dplus,Dminus

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonCharge(external_fluxes)
        return H

    def modeImpedance(self):
        Cn_,Ln_,basis = self.Cn_,self.Ln_,self.basis
        impedance = [sqrt(Cn_[i,i]/Ln_[i,i]) for i in range(len(basis))]
        return impedance

    def oscillatorInitialization(self):
        Q,F = self.Q,self.F
        N = self.Nn
        QQ,FF = dict(),dict()
        for i in range(N):
            for j in range(N):
                QQ[(i,j)] = self.backend.modeProduct(Q,i,Q,j)
                FF[(i,j)] = self.backend.modeProduct(F,i,F,j)
        return QQ,FF

    def josephsonInitialization(self):
        N = self.basisSize()
        H = self.backend.null(N)
        Dplus,Dminus = self.D_plus,self.D_minus
        Jplus,Jminus = dict(),dict()
        edges,Ej = self.josephsonComponents()
        for (u,v,key),E in zip(edges,Ej):
            edge = u,v
            i,j = self.nodes_[u],self.nodes_[v]
            if i<0 or j<0 :
                # grounded josephson
                i = max(i,j)
                Jplus[edge] = self.backend.basisProduct(Dplus,[i])
                Jminus[edge] = self.backend.basisProduct(Dminus,[i])
            else:
                Jplus[edge] = self.backend.crossBasisProduct(Dplus,Dminus,i,j)
                Jminus[edge] = self.backend.crossBasisProduct(Dplus,Dminus,j,i)

        return Jplus,Jminus

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianLC(self):
        Cn_,Ln_ = self.Cn_,self.Ln_
        H = self.null.clone()
        H = coeffProduct(H,Cn_,self.QQ)
        H = coeffProduct(H,Ln_,self.FF)

        return H/2
    
    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        H = self.null.clone()
        for (u,v,key),E in zip(edges,Ej):
            edge = u,v
            flux = self.loopFlux(u,v,key,external_fluxes)
            H -= E*(self.Jplus[edge]*phase(flux) + self.Jminus[edge]*phase(-flux))

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

class Mixed(Circuit):
    def __init__(self,network,basis,sparse=True,pairs=dict()):
        super().__init__(network,basis,sparse,pairs,device)

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

if __name__=='__main__':
    import models
    from DiSuQ.Torch.circuit import Kerman,Charge
    torch.set_num_threads(12)
    basis = [6]*3 ; Rep = Charge
    basis = {'O':[10],'I':[],'J':[4,4]} ; Rep = Kerman
    circuit = models.shuntedQubit(Rep,basis,sparse=True)
    flux_manifold = zip(arange(0,1,.01))
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson
    eige = circuit.circuitEnergy(H_LC,H_J,{'I':tensor(.225)})
    import ipdb; ipdb.set_trace()
