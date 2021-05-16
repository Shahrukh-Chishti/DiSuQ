import numpy
import numpy as np
from scipy.optimize import minimize as sp_minimize
from utils import *

def meritTargetSpectrum(spectrum,target,C,J,circuit_bounds):

    # Calculated and target spectrum
    #spectrum   = np.array(circuit['measurements']['eigen_spectrum'])
    #targetspec = np.array(merit_options['target_spectrum'])

    # Calculate loss from mean square of spectra difference
    loss_flux = np.mean((spectrum[:,1:3]-target[:,1:3])**2)
    loss = loss_flux

    #C,J = matrixArray(C),matrixArray(J) # convert from matrix to array

    # Symmetry enforcement for 2-node circuits without linear inductances
    C_norm = (C - circuit_bounds['C']['low']) / circuit_bounds['C']['high']
    J_norm = (J  - circuit_bounds['J']['low']) / circuit_bounds['J']['high']

    loss_symmetry = np.abs(C_norm[0] - C_norm[2]) + np.abs(J_norm[0] - J_norm[2])
    loss += 100 * loss_symmetry

    # Apply squashing function
    loss = np.log10(loss)
    return loss

def eigen2Node(C,J,phiExt_fix=0,n=6):
    ## C,J,L expected in matrix format
    C += 1/26.6 * J #add junction capacitance

    # Capacitance matrix C (not to be confused with Capacitance connectivity matrix C)
    C = np.diag(np.sum(C, axis=0)) + np.diag(np.diag(C)) - C
    C = C * 10.**(-15) #convert fF -> F

    # Capacitive (kinetic) part of Hamiltonian
    e = 1.60217662 * 10**(-19) #elementary charge
    h = 6.62607004 * 10**(-34) #Planck constant
    T = np.zeros( ((2*n+1)**len(C), (2*n+1)**len(C)) ) #kinetic part of Hamiltonian
    Cinv = np.linalg.inv(C)
    I = np.eye(2*n+1) #identity matrix
    Q = np.diag(np.arange(-n,n+1)) #Charge operator
    Q1 = Q #+ qExt_fix[0]*I
    Q2 = Q #+ qExt_fix[1]*I
    # More simple construction specific to flux qubit
    T += 0.5*Cinv[0,0] * np.kron(Q1.dot(Q1), I)
    T += 0.5*Cinv[1,1] * np.kron(I, Q2.dot(Q2))
    T += Cinv[0,1] * np.kron(Q1, Q2)
    T *= 4*e**2/h

    # Josephson potential part (specific to flux qubit)
    Jmat = J * 10.**9 #convert GHz -> Hz
    U = np.zeros(((2*n+1)**len(C),(2*n+1)**len(C))) #potential part of Hamiltonian
    Dp = np.diag(np.ones((2*n+1)-1), k=1)
    Dm = np.diag(np.ones((2*n+1)-1), k=-1)
    # Add displacement operator terms that were obtained from cosines
    U = U - Jmat[0,0]/2 * np.kron((Dp + Dm),I)
    U = U - Jmat[1,1]/2 * np.kron(I, (Dp + Dm))
    U = U - Jmat[0,1]/2 * ( np.exp(-2*np.pi*1j*phiExt_fix) * np.kron(Dp,Dm) + np.exp(2*np.pi*1j*phiExt_fix) * np.kron(Dm,Dp) )

    # Assemble Hamiltonian
    H = T + U

    evals = np.linalg.eigh(H)[0]
    evals /= 1e9 #convert to GHz

    return evals

def lossBuilder(target_spectrum,phiExt_sweep,circuit_bounds):
    def loss(Circuit):
        C,J = circuitUnwrap(Circuit)
        Cmat,Jmat = arrayMatrix(C),arrayMatrix(J)
        spec = []
        for phi in phiExt_sweep:
            spec.append(eigen2Node(Cmat,Jmat,phiExt_fix=phi))
        spec = np.array(spec)
        e0 = np.array([spec[i][0] for i in range(len(spec))])
        spec = (spec.T - e0).T
        loss = meritTargetSpectrum(spec,target_spectrum,C,J,circuit_bounds)
        return loss
    return loss

def defineComponentBounds(circuit_bounds,N):
    bounds = [(circuit_bounds['C']['low'],circuit_bounds['C']['high'])]*N
    bounds += [(circuit_bounds['J']['low'],circuit_bounds['J']['high'])]*N
    if 'L' in circuit_bounds:
        bounds += [(circuit_bounds['L']['low'],circuit_bounds['L']['high'])]*N
    return bounds
