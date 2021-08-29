import numpy,utils
import numpy as np
from numpy import cos,sin,pi,exp
from numpy.linalg import det

im = 1.0j

def hamiltonianEnergy(H):
    eigenenergies = numpy.real(numpy.linalg.eigvals(H))
    eigenenergies.sort()
    return eigenenergies

def phase(phi):
    # phi = flux/flux_quanta
    return exp(im*2*pi*phi)

def hamiltonian(n,flux):
    Dp = np.diag(np.ones((2*n+1)-1), k=-1)
    Dm = np.diag(np.ones((2*n+1)-1), k=1)
    I = np.eye(2*n+1) #identity matrix
    Jplus = numpy.kron(Dp,Dm)
    Jminus = numpy.kron(Dm,Dp)
    Jplus = numpy.kron(Jplus,I)
    Jminus = numpy.kron(Jminus,I)
    H = Jplus*phase(flux) + Jminus*phase(-flux)
    #H += Jplus + Jminus
    #H += numpy.kron(Dp,I) + numpy.kron(Dm,I)
    #H += numpy.kron(numpy.kron(Dp+Dm,I),I)
    #H += numpy.kron(numpy.kron(I,Dp+Dm),I)
    #H += numpy.kron(numpy.kron(I,I),I)
    return H

def spectrumFlux(flux_manifold):
    spectrum = []
    for flux in flux_manifold:
        H = hamiltonian(3,flux)
        print(det(H))
        ground_energy = hamiltonianEnergy(H)[0]
        spectrum.append(ground_energy)
    return spectrum

if __name__ == "__main__":
    flux_manifold = numpy.arange(0,1,.01)
    spectrum = spectrumFlux(flux_manifold)
    utils.plotCompare(numpy.arange(0,1,.01),{'I':spectrum})
