import numpy,uuid
from numpy import cos,sin,pi,exp,sqrt,array
from numpy.linalg import det,norm
from scipy.linalg import expm
numpy.set_printoptions(precision=2)

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

def diagonalisation(M):
    eig,vec = numpy.linalg.eig(M)
    indices = numpy.argsort(eig)
    D = numpy.asarray(vec[:,indices])
    return D

def unitaryTransformation(M,U):
    M = U.conj().T @ M @ U
    return M

def identity(n):
    return numpy.eye(n)

def basisQo(n,impedance):
    Qo = numpy.arange(1,n)
    Qo = numpy.sqrt(Qo)
    Qo = -numpy.diag(Qo,k=1) + numpy.diag(Qo,k=-1)
    return Qo*im*numpy.sqrt(1/2/pi/impedance)

def basisFo(n,impedance):
    Po = numpy.arange(1,n)
    Po = numpy.sqrt(Po)
    Po = numpy.diag(Po,k=1) + numpy.diag(Po,k=-1)
    return Po*numpy.sqrt(impedance/2/pi)

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
    charge = numpy.linspace(n,-n,2*n+1,dtype=int)
    return charge

def fluxStates(N_flux,n_flux=1):
    flux = numpy.linspace(n_flux,-n_flux,N_flux)
    return flux/N_flux

def transformationMatrix(n_charge,N_flux,n_flux=1):
    charge_states = numpy.linspace(n_charge,-n_charge,2*n_charge+1,dtype=numpy.complex128)
    flux_states = numpy.linspace(n_flux,-n_flux,N_flux,dtype=numpy.complex128)

    T = numpy.matrix(flux_states).T @ numpy.matrix(charge_states)
    T *= 2*pi*im/N_flux
    T = exp(T)/sqrt(N_flux)
    return array(T)

def basisQq(n):
    # charge basis
    charge = chargeStates(n)
    Q = numpy.zeros((len(charge),len(charge)),numpy.complex128)
    numpy.fill_diagonal(Q,charge)
    return Q * 2

def basisFq(n):
    # charge basis
    N = 2*n+1
    P = numpy.zeros((N,N),dtype=numpy.complex128)
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
    F = numpy.zeros((len(flux),len(flux)),numpy.complex128)
    numpy.fill_diagonal(F,flux)
    return F

def basisQf(n):
    F = basisFf(n)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = numpy.ones((2*n+1)-1,dtype=numpy.complex)
    D = numpy.diag(diagonal,k=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = numpy.ones((2*n+1)-1,dtype=numpy.complex)
    D = numpy.diag(diagonal,k=1)
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
    eig,vec = numpy.linalg.eigh(H)
    indices = numpy.argsort(eig)
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
    Po = basisPo(5,1)
    F = fluxCharge(2,1)
    import ipdb;ipdb.set_trace()
