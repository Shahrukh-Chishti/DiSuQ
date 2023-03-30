import uuid,numpy
from numpy import sqrt as sqroot,pi,prod

from torch import tensor,norm,abs,ones,zeros,zeros_like,argsort
from torch import linspace,arange,diagonal,diag,sqrt,eye
from torch.linalg import det,inv,eig as eigsolve
from torch import matrix_exp as expm,exp,outer
from torch import sin,cos,sigmoid
from torch import cfloat as complex, float32 as float

im = 1.0j
root2 = sqroot(2)
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
hbar = h/2/pi
flux_quanta = h/2/e
Z0 = h/4/e/e
Z0 = flux_quanta / 2 / e

# upper limit of circuit elements
# Energy units in GHz
J0,C0,L0 = 1200,2500,1200

def null(H):
    def empty(*args):
        return H*0
    return empty

# conversion SI and Natural units

def capSINat(cap):
    return cap/(e**2/h/1e9)

def capNatSI(cap):
    return (e**2/1e9/h)*cap

def indSINat(ind):
    return ind/(flux_quanta**2/h/1e9)

def indSINat(ind):
    return ind*(flux_quanta**2/h/1e9)

# conversion Natural and Energy units

def capEnergy(cap):
    # natural <-> energy
    return 1. / 2 / cap # GHz

def indEnergy(ind):
    # natural <-> energy
    return 1. / 4 / pi**2 / ind # GHz

# natural -- involution -- energy

# conversion SI to Energy units 

def capE(cap):
    return 1 / 2 / cap * e * e / h / 1e9

def indE(ind):
    return 1 / 4 / pi**2 / ind * flux_quanta**2 / h / 1e9

def sigmoidInverse(x):
    return -numpy.log(1/x -1)

def normalize(state,square=True):
    state = abs(state)
    norm_state = norm(state)
    state = state/norm_state
    if square:
        state = abs(state)**2
    return state

def diagonalisation(M,reverse=False):
    eig,vec = eigsolve(M)
    if reverse:
        eig = -eig
    indices = argsort(eig.real)
    D = vec[:,indices].clone().detach()
    return D

# components are standardized with natural & energy units

def classComponents(component):
    var,energy,phys = None,None,None
    if component.__class__ == C :
        var = component.cap
        energy = component.energy()
        phys = component.capacitance()
        bound = component.C0
        
    elif component.__class__ == L :
        var = component.cap
        energy = component.energy()
        phys = component.capacitance()
        bound = component.L0
        
    elif component.__class__ == J :
        var = component.cap
        energy = component.energy()
        phys = component.capacitance()
        bound = component.J0
        
    return var,energy,phys

class Elements:
    def __init__(self,plus,minus,ID=None):
        self.plus = plus
        self.minus = minus
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID = ID

class J(Elements):
    def __init__(self,plus,minus,Ej,ID=None,J0=J0):
        super().__init__(plus,minus,ID)
        self.J0 = J0
        self.initJunc(Ej) # Ej[GHz]
        
    def initJunc(self,Ej):
        self.jo = tensor(sigmoidInverse(Ej/self.J0),dtype=float,requires_grad=True)

    def energy(self):
        return sigmoid(self.jo)/1.0 * self.J0 # GHz

class C(Elements):
    def __init__(self,plus,minus,Ec,ID=None,C0=C0):
        super().__init__(plus,minus,ID)
        self.C0 = C0
        self.initCap(Ec) # Ec[GHz]
        
    def initCap(self,Ec):
        self.cap = tensor(sigmoidInverse(Ec/self.C0),dtype=float,requires_grad=True)

    def energy(self):
        return sigmoid(self.cap)*self.C0 # GHz

    def capacitance(self):
        return capEnergy(self.energy()) # he9/e/e : natural unit

class L(Elements):
    def __init__(self,plus,minus,El,ID=None,external=False,L0=L0):
        super().__init__(plus,minus,ID)
        self.L0 = L0
        self.external = external # duplication : doesn't requires_grad
        self.initInd(El) # El[GHz]
        
    def initInd(self,El):
        self.ind = tensor(sigmoidInverse(El/self.L0),dtype=float,requires_grad=True)

    def energy(self):
        return sigmoid(self.ind)*self.L0 # GHz

    def inductance(self):
        return indEnergy(self.energy()) # 4e9 e^2/h : natural unit

if __name__=='__main__':
    Qo = basisQo(30,tensor(4))
    Fq = basisFq(30)
    Qf = basisQf(30)
    S1 = torch.sparse_coo_tensor([[0],[0]],[1.0],[3,3])
    S2 = torch.sparse_coo_tensor([[0],[0]],[4.0],[3,3])
    s1 = zeros(3,3);s1[0,0]=1;s1[1,2]=234
    s2 = zeros(3,3);s2[0,0]=4
    Sp1 = sparsify(s1)
    Sp2 = sparsify(s2)
    import ipdb;ipdb.set_trace()
