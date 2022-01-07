import numpy,uuid,torch
from numpy import cos,sin,pi,array,argsort,asarray,eye
from numpy import linspace,matrix
from numpy import dot,real,prod,arange
from numpy.linalg import det,norm,eig as eigsolve,eigh,inv,eigvals,matrix_rank

from torch.linalg import det,inv
from torch import matrix_exp as expm,kron,diagonal,diag,sqrt,vstack
from torch import eye,tensor,matmul as mul,zeros,zeros_like,nonzero,exp
from torch import ones,complex128

numpy.set_printoptions(precision=3)

im = 1.0j
root2 = numpy.sqrt(2)
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
hbar = h/2/pi
flux_quanta = h/2/e
Z0 = h/4/e/e
Z0 = flux_quanta / 2 / e

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
    indices = argsort(eig)
    D = asarray(vec[:,indices])
    return D

def unitaryTransformation(M,U):
    M = mul(U.conj().T, mul(M, U))
    return M

def identity(n):
    return eye(n)

def sparsify(T):
    indices = nonzero(T,as_tuple=True)
    shape = T.shape
    values = T[indices]
    indices = vstack(indices)
    return torch.sparse_coo_tensor(indices,values,shape)

def basisQo(n,impedance):
    Qo = torch.arange(1,n)
    Qo = sqrt(Qo)
    Qo = -diag(Qo,diagonal=1) + diag(Qo,diagonal=-1)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance):
    Po = torch.arange(1,n)
    Po = sqrt(Po)
    Po = diag(Po,diagonal=1) + diag(Po,diagonal=-1)
    return Po*sqrt(impedance/2/pi)

def fluxFlux(n,impedance):
    N = 2*n+1
    Po = basisPo(N,impedance)
    D = diagonalisation(Po)
    Pp = unitaryTransformation(Po,D)
    return Pp

def chargeFlux(n,impedance):
    N = 2*n+1
    Po = basisPo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Po)
    Qp = unitaryTransformation(Qo,D)
    return Qp

def chargeCharge(n,impedance):
    N = 2*n+1
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Qq = unitaryTransformation(Qo,D)
    return Qq

def fluxCharge(n,impedance):
    N = 2*n+1
    Po = basisPo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Pq = unitaryTransformation(Po,D)
    return Pq

def chargeStates(n):
    charge = linspace(n,-n,2*n+1)
    return charge

def fluxStates(N_flux,n_flux=1):
    flux = linspace(n_flux,-n_flux,N_flux)
    return flux/N_flux

def transformationMatrix(n_charge,N_flux,n_flux=1):
    charge_states = chargeStates(n_charge)
    flux_states = fluxStates(N_flux,n_flux)

    T = matrix(flux_states).T @ matrix(charge_states)
    T = tensor(T,dtype=complex128)
    T *= 2*pi*im
    T = exp(T)/numpy.sqrt(N_flux)
    return T

def basisQq(n):
    # charge basis
    charge = chargeStates(n)
    Q = diag(tensor(charge,dtype=complex128))
    return Q * 2

def basisFq(n):
    # charge basis
    N = 2*n+1
    P = zeros((N,N),dtype=complex128)
    charge = chargeStates(n)
    for q in charge:
        for p in charge:
            if not p==q:
                P[q,p] = (-(n+1)*sin(2*pi*(q-p)*n/N) + n*sin(2*pi*(q-p)*(n+1)/N))
                P[q,p] /= -im*N*(1-cos(2*pi*(q-p)/N))*N
    return P

def basisFq(n):
    Q = basisQq(n)
    U = transformationMatrix(n,2*n+1,n)
    return U@Q@U.conj().T/(2.0*n+1.0)/2.0

def basisFf(n):
    flux = fluxStates(2*n+1,n)
    F = diag(tensor(flux))
    return F

def basisQf(n):
    F = basisFf(n)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex128)
    D = diag(diagonal,k=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex128)
    D = diag(diagonal,k=1)
    return D

def displacementCharge(n,a):
    D = basisFq(n)
    D = expm(im*2*pi*a*D)
    return D

def displacementOscillator(n,z,a):
    D = basisFo(n,z)
    D = expm(im*2*pi*a*D)
    return D

def displacementFlux(n,a):
    D = basisFf(n)
    D = expm(im*2*pi*a*D)
    return D

def wavefunction(H,level=[0]):
    eig,vec = eigh(H)
    indices = argsort(eig)
    states = vec.T[indices[level]]
    return states

def null(*args):
    return 0

class Elements:
    def __init__(self,plus,minus,ID=None):
        self.plus = plus
        self.minus = minus
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID = ID

class J(Elements):
    def __init__(self,plus,minus,Ej,ID=None):
        super().__init__(plus,minus,ID)
        self.energy = tensor(Ej/1.0,requires_grad=True)

class C(Elements):
    def __init__(self,plus,minus,Ec,ID=None):
        super().__init__(plus,minus,ID)
        self.capacitance = tensor(1/Ec/2,requires_grad=True)

class L(Elements):
    def __init__(self,plus,minus,El,ID=None,external=False):
        super().__init__(plus,minus,ID)
        self.inductance = tensor(1/El/4/pi/pi,requires_grad=True)
        self.external = external

if __name__=='__main__':
    Qo = basisQo(30,tensor(4))
    Fq = basisFq(30)
    S1 = torch.sparse_coo_tensor([[0],[0]],[1.0],[3,3])
    S2 = torch.sparse_coo_tensor([[0],[0]],[4.0],[3,3])
    s1 = zeros(3,3);s1[0,0]=1;s1[1,2]=234
    s2 = zeros(3,3);s2[0,0]=4
    Sp1 = sparsify(s1)
    Sp2 = sparsify(s2)
    import ipdb;ipdb.set_trace()
