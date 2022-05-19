import numpy,uuid
from numpy import cos,sin,pi,exp,sqrt,array,arange,argsort,asarray,eye
from numpy import diag,linspace,matrix,complex128,zeros,fill_diagonal,ones
from numpy import dot,zeros_like,real,prod,diagonal
from numpy.linalg import det,norm,eig as eigsolve,eigh,inv,eigvals,matrix_rank
from scipy.linalg import expm
numpy.set_printoptions(precision=2)

im = 1.0j
root2 = sqrt(2)
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
hbar = h/2/pi
flux_quanta = h/2/e
Z0 = h/4/e/e
Z0 = flux_quanta / 2 / e

def sparsity(M,tol=10):
    M = numpy.around(M,tol)
    return 1-numpy.count_nonzero(M)/prod(M.shape)

def isHermitian(M,tol=10):
    M = numpy.around(M,tol)
    return (M==M.conjugate().T).all()

def normalize(state,square=True):
    state = abs(state)
    norm_state = norm(state)
    state = state/norm_state
    if square:
        state = abs(state)**2
    return state

def diagonalisation(M,inverse=False):
    eig,vec = eigsolve(M)
    if inverse:
        eig = -eig
    indices = argsort(eig)
    D = asarray(vec[:,indices])
    return D

def unitaryTransformation(M,U):
    M = U.conj().T @ M @ U
    return M

def identity(n):
    return eye(n)

def basisQo(n,impedance):
    Qo = arange(1,n)
    Qo = sqrt(Qo)
    Qo = -diag(Qo,k=1) + diag(Qo,k=-1)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance):
    Fo = arange(1,n)
    Fo = sqrt(Fo)
    Fo = diag(Fo,k=1) + diag(Fo,k=-1)
    return Fo*sqrt(impedance/2/pi)

def fluxFlux(n,impedance):
    N = 2*n+1
    Fo = basisFo(N,impedance)
    D = diagonalisation(Fo)
    Pp = unitaryTransformation(Fo,D)
    return Pp

def chargeFlux(n,impedance):
    N = 2*n+1
    Fo = basisFo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Fo)
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
    Fo = basisFo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Pq = unitaryTransformation(Fo,D)
    return Pq

def chargeStates(n):
    charge = linspace(n,-n,2*n+1,dtype=int)
    return charge

def fluxStates(N_flux,n_flux=1):
    flux = linspace(n_flux,-n_flux,N_flux)
    return flux/N_flux

def transformationMatrix(n_charge,N_flux,n_flux=1):
    charge_states = linspace(n_charge,-n_charge,2*n_charge+1,dtype=complex128)
    flux_states = linspace(n_flux,-n_flux,N_flux,dtype=complex128)

    T = matrix(flux_states).T @ matrix(charge_states)
    T *= 2*pi*im/N_flux
    T = exp(T)/sqrt(N_flux)
    return array(T)

def basisQq(n):
    # charge basis
    charge = chargeStates(n)
    Q = zeros((len(charge),len(charge)),complex128)
    fill_diagonal(Q,charge)
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
    return U@Q@U.conj().T/(2*n+1)/2

def basisFf(n):
    flux = fluxStates(2*n+1,n)
    F = zeros((len(flux),len(flux)),complex128)
    fill_diagonal(F,flux)
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
        self.energy = Ej

class C(Elements):
    def __init__(self,plus,minus,Ec,ID=None):
        super().__init__(plus,minus,ID)
        self.capacitance = 1/Ec/2

class L(Elements):
    def __init__(self,plus,minus,El,ID=None,external=False):
        super().__init__(plus,minus,ID)
        self.inductance = 1/El/4/pi/pi
        self.external = external

if __name__=='__main__':
    Fo = basisFo(5,1)
    F = fluxCharge(2,1)
    import ipdb;ipdb.set_trace()
