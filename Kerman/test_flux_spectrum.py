import numpy,utils,sys
import numpy as np
from numpy.linalg import det,norm
from components import *

def hamiltonianEnergy(H):
    eigenenergies = numpy.real(numpy.linalg.eigvals(H))
    eigenenergies.sort()
    return eigenenergies

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def kronecker(modes):
    product = 1
    for matrix in modes:
        product = numpy.kron(product,matrix)
    return product

def hamiltonianKerman(n,flux):
    Dp = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=-1)
    Dm = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=1)
    I = np.eye(2*n+1) #identity matrix
    P = basisPj(n)
    P2 = numpy.dot(P,P)
    H = kronecker([I,Dp,Dm])*phase(flux) + kronecker([I,Dm,Dp])*phase(-flux)
    H += kronecker([Dp,Dm,I]) + kronecker([Dm,Dp,I])
    H += kronecker([Dp,I,I]) + kronecker([Dm,I,I])
    H -= kronecker([I,I,P2])
    return H

def josephsonE(n,flux):
    Dp = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=-1)
    Dm = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=1)
    I = np.eye(2*n+1) #identity matrix

    H = kronecker([I,Dp,Dm])*phase(flux) + kronecker([I,Dm,Dp])*phase(-flux)
    H += kronecker([Dp,Dm,I]) + kronecker([Dm,Dp,I])
    H += kronecker([Dp,I,I]) + kronecker([Dm,I,I])
    return -H * 1e9 / 2

def hamiltonianSCILLA(n,flux):
    Dp = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=-1)
    Dm = np.diag(np.ones((2*n+1)-1,dtype=np.complex128), k=1)
    I = np.eye(2*n+1) #identity matrix
    Jplus = numpy.kron(Dp,Dm)
    Jminus = numpy.kron(Dm,Dp)
    #Jplus = numpy.kron(Jplus)
    #Jminus = numpy.kron(Jminus)
    H = Jplus + Jminus
    #H += Jplus + Jminus
    #H += numpy.kron(Dp,I) + numpy.kron(Dm,I)
    H += numpy.kron(Dp+Dm,I)
    H += numpy.kron(I,Dp)*phase(flux)
    H += numpy.kron(I,Dm)*phase(-flux)
    #H += numpy.kron(numpy.kron(I,I),I)
    return H

def spectrumFlux(flux_manifold):
    spectrum = []
    for flux in flux_manifold:
        H = hamiltonianKerman(1,flux)
        ground_energy = hamiltonianEnergy(H)[0]
        spectrum.append(ground_energy)
    return spectrum

def fluxBasisLC(n):
    N = 2*n + 1
    L,C = 10e-6,1e-18
    flux_max = 4*pi*flux_quanta
    flux = numpy.linspace(-flux_max,flux_max,N)
    delta = abs(flux[1]-flux[0])
    flux = numpy.diag(flux)
    flux2 = numpy.dot(flux,flux)

    charge = -numpy.diag(numpy.ones(N-1),k=-1)
    charge += numpy.diag(numpy.ones(N-1),k=1)
    charge = hbar/2/delta * charge
    charge2 = -numpy.dot(charge,charge)
    #import ipdb; ipdb.set_trace()

    H = (charge2/C + flux2/L)/2/h
    return H

def fluxBasisLC(n):
    N = 2*n + 1
    L,C = 1000e-12,4000e-15
    impedance = numpy.sqrt(L/C)
    flux = fluxFlux(N,impedance)
    flux2 = numpy.dot(flux,flux)

    charge = chargeFlux(N,impedance)
    charge2 = numpy.dot(charge,charge)

    H = charge2/C/2 + flux2/L/2
    return H

def chargeBasisLC(n):
    N = 2*n + 1
    L,C = 1000e-12,4000e-15
    impedance = numpy.sqrt(L/C)
    flux = fluxCharge(N,impedance)
    flux2 = numpy.dot(flux,flux)

    charge = chargeCharge(N,impedance)
    charge2 = numpy.dot(charge,charge)

    H = charge2/C/2 + flux2/L/2
    return H

def oscillatorLC(n):
    L,C = 1000e-12,4000e-15 # SI units
    L *= 1e9 # GHz
    C *= 1e9 # GHz
    print(numpy.sqrt(L/C))
    # energy  units
    L = L * 2 * h / flux_quanta**2
    C = C * 2 * h / 4 / e / e
    print(C)
    print(L)
    impedance = numpy.sqrt(L/C)
    print(impedance*Z0)
    flux = basisPo(n,impedance)
    flux2 = numpy.dot(flux,flux)

    charge = basisQo(n,impedance)
    charge2 = numpy.dot(charge,charge)

    H = charge2/C/2 + flux2/L/2
    return H

def basisTransformation(n):
    Po = numpy.arange(1,n)
    Po = numpy.sqrt(Po)
    Po = numpy.diag(Po,k=1) + numpy.diag(Po,k=-1)

    Qo = numpy.arange(1,n)
    Qo = numpy.sqrt(Qo)
    Qo = -numpy.diag(Qo,k=1) + numpy.diag(Qo,k=-1)
    Qo = numpy.asarray(Qo,dtype=numpy.complex128)
    Qo *= im

    import ipdb; ipdb.set_trace()
    D = diagonalisation(Po)
    Pp = unitaryTransformation(Po,D)
    Qp = unitaryTransformation(Qo,D)

    G = diagonalisation(Qo)
    Qq = unitaryTransformation(Qo,G)
    Pq = unitaryTransformation(Po,G)

if __name__ == "__main__":
    H = oscillatorLC(15)
    E = hamiltonianEnergy(H[:-1,:-1])# / 1e9
    print(numpy.diff(E))
    import ipdb; ipdb.set_trace()
    flux_manifold = numpy.arange(0,3,.01)
    spectrum = spectrumFlux(flux_manifold)
    utils.plotCompare(numpy.arange(0,3,.01),{'I':spectrum})
