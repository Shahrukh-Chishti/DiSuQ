from torch import kron
from components import *

""" Operator Computation """

def modeTensorProduct(pre,M,post):
    """
        extend mode to full system basis
        sequentially process duplication
    """
    H = identity(1)
    for dim in pre:
        H = kron(H,identity(dim))
    H = kron(H,M)
    for dim in post:
        H = kron(H,identity(dim))
    return H

def crossBasisProduct(A,B,a,b):
    assert len(A)==len(B)
    n = len(A)
    product = identity(1)
    for i in range(n):
        if i==a:
            product = kron(product,A[i])
        elif i==b:
            product = kron(product,B[i])
        else:
            product = kron(product,identity(len(A[i])))
    return product

def basisProduct(O,indices=None):
    n = len(O)
    B = identity(1)
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
        B : list : basis operators
        A : list : basis operators(transpose)
        cross_mode : indicates if A!=B, assumed ordering : AxB
        returns : prod(nA) x prod(nB) mode state Hamiltonian matrix
    """
    shape = prod([len(a) for a in A])
    H = null(shape)
    a,b = mode
    nA,nB = M.shape
    for i in range(nA):
        for j in range(nB):
            left = basisProduct(A,[i+a])
            right = basisProduct(B,[j+b])
            H += M[i,j]*mul(left,right)

    return H

""" Operator Objects """

def unitaryTransformation(M,U):
    M = U.conj().T@ M@ U
    return M

def identity(n):
    return eye(n)

def null(N=1,dtype=complex):
    return zeros(N,N,dtype=complex)

def mul(A,B):
    return A@B

def basisQo(n,impedance):
    Qo = arange(1,n)
    Qo = sqrt(Qo)
    Qo = -diag(Qo,diagonal=1) + diag(Qo,diagonal=-1)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance):
    Po = arange(1,n)
    Po = sqrt(Po)
    Po = diag(Po,diagonal=1) + diag(Po,diagonal=-1)
    return Po*sqrt(impedance/2/pi)

def fluxFlux(n,impedance):
    N = 2*n+1
    Po = basisFo(N,impedance)
    D = diagonalisation(Po)
    Pp = unitaryTransformation(Po,D)
    return Pp

def chargeFlux(n,impedance):
    N = 2*n+1
    Po = basisFo(N,impedance)
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
    Po = basisFo(N,impedance)
    Qo = basisQo(N,impedance)
    D = diagonalisation(Qo)
    Pq = unitaryTransformation(Po,D)
    return Pq

def chargeStates(n,dtype=int):
    charge = linspace(n,-n,2*n+1,dtype=dtype)
    return charge

def fluxStates(N_flux,n_flux=1,dtype=complex):
    flux = linspace(n_flux,-n_flux,N_flux,dtype=dtype)
    return flux/N_flux

def transformationMatrix(n_charge,N_flux,n_flux=1):
    charge_states = chargeStates(n_charge,complex)
    flux_states = fluxStates(N_flux,n_flux,complex)*N_flux

    T = outer(flux_states,charge_states)
    T *= 2*pi*im/N_flux
    T = exp(T)/sqroot(N_flux)
    return T

def basisQq(n):
    # charge basis
    charge = chargeStates(n,complex)
    Q = diag(charge.clone().detach())
    return Q * 2

def basisFq(n):
    # charge basis
    N = 2*n+1
    P = zeros((N,N),dtype=complex)
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
    flux = fluxStates(2*n+1,n,complex)
    F = diag(flux)
    return F

def basisQf(n):
    F = basisFf(n).to(complex)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex)
    D = diag(diagonal,diagonal=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex)
    D = diag(diagonal,diagonal=1)
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

if __name__=='__main__':
    Qo = basisQo(30,tensor(4))
    Fq = basisFq(30)
    Qf = basisQf(30)

    import ipdb;ipdb.set_trace()
