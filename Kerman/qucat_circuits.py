import numpy
import numpy as np
from qucat import Network,J,C,L,R
from scipy.constants import pi,hbar, h, e

def transmon():
    cir = [J(0,1,1000e-9),C(0,1,50000e-15)]
    return Network(cir)

def oscillatorLC():
    cir = [L(0,1,1000e-12),C(0,1,4000e-15)]
    return Network(cir)

def Lj(phi):
    # maximum Josephson energy
    Ejmax = 6.5e9
    # junction asymmetry
    d = 0.0769
    # flux to Josephson energy
    Ej = Ejmax*np.cos(pi*phi) *np.sqrt(1+d**2 *np.tan(pi*phi)**2)
    # Josephson energy to inductance
    return (hbar/2/e)**2/(Ej*h)

if __name__ == '__main__':
    circuit = oscillatorLC()
    H = circuit.hamiltonian()
    ee = H.eigenenergies()
    print(ee[1])
