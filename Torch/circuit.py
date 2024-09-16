import networkx,copy,torch
from contextlib import nullcontext
import DiSuQ.Torch.dense as Dense
import DiSuQ.Torch.sparse as Sparse
from torch import exp,det,tensor,arange,zeros,sqrt,diagonal,lobpcg,rand,eye
from torch.linalg import eigvalsh,inv,eigh
from DiSuQ.Torch.components import diagonalisation,null,J,L,C,im,pi,complex,float
from numpy.linalg import matrix_rank
from numpy import prod,sort
from torch import nn

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

def hamiltonianEnergy(H):
    eigenenergies = eigvalsh(H)
    return eigenenergies

### Circuit Module
class Circuit(nn.Module):
    """
        * no external fluxes must be in parallel : redundant
        * no LC component must be in parallel : redundant
        * only Josephson elements allowed in parallel
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
        # basis Size undefined in Root class
        self.null = self.backend.null(self.basisSize(),device=self.device)

        self.spectrum_limit = 4
        self.ii_limit = 3
        self.vectors_calc = False
        # nn.Module : default function call in optimization
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
        # control data : array/list of Tensor/s
        control = self.controlData(control)
        algo = None
        with torch.inference_mode() if self.grad_calc is False else nullcontext() as null:
            H = self.circuitHamiltonian(control)
            spectrum = eigvalsh(H)[:self.spectrum_limit]
            states = None
            if self.vectors_calc:
                states = self.eigenVectors(H,spectrum)
        return spectrum,states

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def eigenVectors(self,H,eigenvalues):
        Vectors = []
        for val in eigenvalues:
            Vectors.append(self.inverseIteration(H,val,iterations=self.ii_limit))
        return Vectors

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def inverseIteration(self,H,val,iterations):
        N = self.N
        vec = rand(N,dtype=complex,requires_grad=True)
        invII = inv(H-val*eye(N))
        for i in range(iterations):
            vec = invII@vec/norm(invII@vec)
        return vec

    def operatorExpectation(self,bra,O,mode,ket):
        basis = self.basis
        O = [O[basis_max] for basis_max in basis]
        O = self.backend.basisProduct(O,[mode])
        return bra.conj()@ O@ ket

    # ToBeDeprecated
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
                eigenenergies = eigvalsh(H)
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

    def fluxScape(self,flux_points,flux_manifold):
        H_LC = self.hamiltonianLC()
        H_J = self.hamiltonianJosephson
        E0,EI = self.spectrumManifold(flux_point,flux_profile,H_LC,H_J,excitation=1)
        E0,EII = self.spectrumManifold(flux_point,flux_profile,H_LC,H_J,excitation=2)
        EI = tensor(EI).detach().numpy()
        EII = tensor(EII).detach().numpy()
        return EI,EII

### Basis Representation

class Kerman(Circuit):
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)
        self.No,self.Ni,self.Nj = self.kermanDistribution()
        self.N = self.basisSize()
        self.R = self.kermanTransform().real
        self.L_,self.C_ = self.modeTransformation()

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
        if Ln_.is_cuda:
            No = matrix_rank(Ln_.cpu().detach().numpy())
        else:
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
        No,Ni,Nj = self.No,self.Ni,self.Nj # self.kermanDistribution()
        O = combination[:No]
        I = combination[No:No+Ni]
        J = combination[No+Ni:]
        
        Z = self.oscillatorImpedance() * 2 # cooper pair factor
        # re-calculation impedance factor with circuit variation
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

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianLC(self):
        """
            basis : {O:(,,,),I:(,,,),J:(,,,)}
        """
        basis = self.basis
        self.Cn_,self.Ln_ = self.componentMatrix()
        self.L_,self.C_ = self.modeTransformation()
        Lo_,C_ = self.kermanComponents()

        #Lo_ = self.Lo_
        Co_,Coi_,Coj_,Ci_,Cij_,Cj_ = C_
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

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=True)
    def hamiltonianJosephson(self,external_fluxes=dict()):
        edges,Ej = self.josephsonComponents()
        N = self.basisSize()
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

    def kermanChargeOffset(self,charge_offset=dict()):
        charge = zeros(self.Nn)
        for node,dQ in charge_offset.items():
            charge[self.nodes_[node]] = dQ
        charge = self.R@charge
        
        No,Ni,Nj = self.kermanDistribution() #No,self.Ni,self.Nj
        
        Qo = [self.backend.basisQo(basis_max) for basis_max in basis['O']]
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
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)

    def hamiltonianLC(self):
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

    def hamiltonianJosephson(self,external_fluxes=dict()):
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

    def potentialOscillator(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        impedance = self.modeImpedance()
        F = [self.backend.basisFo(basis_max,impedance[index]) for index,basis_max in enumerate(basis)]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonOscillator(external_fluxes)
        return H

class Flux(Circuit):
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)

    def hamiltonianLC(self):
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

    def hamiltonianJosephson(self,external_fluxes=dict()):
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

    def potentialFlux(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFf(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonFlux(external_fluxes)
        return H

class Charge(Circuit):
    def __init__(self,network,control_iD,basis,sparse=True,pairs=dict(),device=None):
        super().__init__(network,control_iD,basis,sparse,pairs,device)
        self.N = self.basisSize()

    def basisSize(self,modes=False):
        N = [2*size+1 for size in self.basis]
        if modes:
            return N
        return prod(N)

    def hamiltonianLC(self):
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

    def hamiltonianJosephson(self,external_fluxes=dict()):
        basis = self.basis
        Dplus = [self.backend.chargeDisplacePlus(basis_max) for basis_max in basis]
        Dminus = [self.backend.chargeDisplaceMinus(basis_max) for basis_max in basis]
        edges,Ej = self.josephsonComponents()
        N = self.basisSize()
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

    def potentialCharge(self,external_fluxes=dict()):
        Ln_ = self.Ln_
        basis = self.basis

        F = [self.backend.basisFq(basis_max) for basis_max in basis]
        H = self.backend.modeMatrixProduct(F,Ln_,F)/2
        H += self.josephsonCharge(external_fluxes)
        return H

class Mixed(Circuit):
    def __init__(self,network,basis,sparse=True,pairs=dict()):
        super().__init__(network,basis,sparse,pairs,device)

    def hamiltonianLC(self,basis,n_trunc):
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

    def hamiltonianJosephson(self,basis,n_trunc):
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
    print(eige)
