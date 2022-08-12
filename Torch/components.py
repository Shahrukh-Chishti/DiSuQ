import uuid,numpy
from numpy import sqrt as sqroot,pi,prod

from torch import tensor,norm,abs,ones,zeros,zeros_like,argsort
from torch import linspace,arange,diagonal,diag,sqrt,eye
from torch.linalg import det,inv,eig as eigsolve
from torch import matrix_exp as expm,exp,outer
from torch import sin,cos,sigmoid
from torch import complex128 as complex, float32 as float

im = 1.0j
root2 = sqroot(2)
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
hbar = h/2/pi
flux_quanta = h/2/e
Z0 = h/4/e/e
Z0 = flux_quanta / 2 / e

# upper limit of circuit elements
J0,C0,L0 = 1000,15000,100

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

class Elements:
    def __init__(self,plus,minus,ID=None):
        self.plus = plus
        self.minus = minus
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID = ID

class J(Elements):
    def __init__(self,plus,minus,jo,ID=None):
        super().__init__(plus,minus,ID)
        self.jo = tensor(sigmoidInverse(jo/J0),requires_grad=True)

    def energy(self):
        return sigmoid(self.jo)/1.0 * J0

class C(Elements):
    def __init__(self,plus,minus,cap,ID=None):
        super().__init__(plus,minus,ID)
        self.cap = tensor(sigmoidInverse(cap/C0),requires_grad=True)

    def capacitance(self):
        return sigmoid(self.cap)*C0

    def energy(self):
        return 1/self.capacitance()/2.0

class L(Elements):
    def __init__(self,plus,minus,ind,ID=None,external=False):
        super().__init__(plus,minus,ID)
        self.ind = tensor(sigmoidInverse(ind/L0),requires_grad=True)
        self.external = external # duplication : doesn't requires_grad

    def inductance(self):
        return sigmoid(self.ind)*L0

    def energy(self):
        return 1/self.inductance()/4/pi/pi


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
