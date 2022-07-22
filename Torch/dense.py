from torch import kron
from components import *

def unitaryTransformation(M,U):
    M = U.conj().T@ M@ U
    return M

def identity(n):
    return eye(n)

def null(N=1,dtype=complex64):
    return zeros(N,N,dtype=complex64)

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
    T = tensor(T,dtype=complex64)
    T *= 2*pi*im/N_flux
    T = expm(T)/sqroot(N_flux)
    return T

def basisQq(n):
    # charge basis
    charge = chargeStates(n)
    Q = diag(charge.clone().detach().to(complex64))
    return Q * 2

def basisFq(n):
    # charge basis
    N = 2*n+1
    P = zeros((N,N),dtype=complex64)
    charge = chargeStates(n)
    for q in charge:
        for p in charge:
            if not p==q:
                P[q,p] = (-(n+1)*sin(2*pi*(q-p)*n/N) + n*sin(2*pi*(q-p)*(n+1)/N))
                P[q,p] /= -im*N*(1-cos(2*pi*(q-p)/N))*N
    return P

def basisFq(n):
    Q = basisQq(n).to(complex64)
    U = transformationMatrix(n,2*n+1,n)
    return U@Q@U.conj().T/(2.0*n+1.0)/2.0

def basisFf(n):
    flux = fluxStates(2*n+1,n)
    F = diag(tensor(flux))
    return F

def basisQf(n):
    F = basisFf(n).to(complex64)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex64)
    D = diag(diagonal,diagonal=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex64)
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
