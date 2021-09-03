import numpy,utils
import numpy as np
from numpy import cos,sin,pi,exp
from numpy.linalg import det
from components import basisPj,im,h,e

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
    H -= kronecker([I,I,P2]) / 100 / h
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

if __name__ == "__main__":
    flux_manifold = numpy.arange(0,3,.01)
    spectrum = spectrumFlux(flux_manifold)
    utils.plotCompare(numpy.arange(0,3,.01),{'I':spectrum})
