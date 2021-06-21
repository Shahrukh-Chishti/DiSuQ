import numpy
import numpy as np
from qucat import Network,J,C,L,R
from scipy.constants import pi,hbar, h, e

def transmon():
    cir = Network([J(0,1,100e-15),C(0,1,500e-15),L(0,1,'Lj')])
    return cir

def Lj(phi):
    # maximum Josephson energy
    Ejmax = 6.5e9
    # junction asymmetry
    d = 0.0769
    # flux to Josephson energy
    Ej = Ejmax*np.cos(pi*phi) *np.sqrt(1+d**2 *np.tan(pi*phi)**2)
    # Josephson energy to inductance
    return (hbar/2/e)**2/(Ej*h)

