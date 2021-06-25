#!/usr/bin/env python 

""" Simulate circuit C from automated discovery results """
import numpy as np 
import numpy.linalg
import scipy as sp
import scipy.sparse.linalg
from qutip import *
import csv
import os
from matplotlib import pyplot as plt

# Identity
def I(n):
	return qeye(2*n+1)

# Charge operator
def Q(n):
	return Qobj( np.diag(np.arange(-n,n+1)) )

# Displacement operator D^+
def Dp(n, k=1):
	if 2*n+1-k > 0:
		d = np.ones(2*n+1-k)
		return Qobj( np.diag(d, k=-k) )
	else:
		return Qobj( np.zeros((2*n+1, 2*n+1)) )

# Displacement operator D^-
def Dm(n, k=1):
	return Dp(n, k).trans()

# phi^2 in terms of displacement operators
def Phi_sq(n):
	Phi_sq = 0
	for k in range(1, 2*n+1):
		Phi_sq += (-1)**k / k**2 * ( Dm(n,k) + Dp(n,k) )
	Phi_sq = 2 * Phi_sq
	return Phi_sq


def spec_circuitC(Carr, L13, L23, lj12, lj22, lj33, phiExt=[[0.0,0.5]], normalized=False, trunc=[6,6,8]):

	# Initialize spectrum
	spec = []

	# Calculate spectrum for swept parameter
	for p in phiExt:
		spec.append( eigs_circuitC(Carr, L13, L23, lj12, lj22, lj33, phiExt_fix=p, trunc=trunc) )
	spec = np.array(spec)

	# Normalize spectrum by ground state if desired
	if normalized:
		e0 = np.array([spec[i][0] for i in range(len(spec))])
		spec = (spec.T - e0).T

	return spec


def eigs_circuitC(Carr, L13, L23, lj12, lj22, lj33, phiExt_fix=[0.0,0.5], trunc=[6,6,8]):

	# Initialization
	n1 = trunc[0]  #node 1 states (oscillator)
	n2 = trunc[1]  #node 2 states (oscillator)
	n3 = trunc[2]  #node 3 states (charge)
	N  = 3  #number of nodes
	
	# Parameters and fundamental constants
	Jc      = 5e-6                   #critical current density in A/um^2
	wJ      = 0.2                    #junction width in um
	Sc      = 60e-15                 #specific capacitance in F/um^2
	e       = 1.60217662 * 10**(-19) #elementary charge
	h       = 6.62607004 * 10**(-34) #Planck constant
	hbar    = h/(2*np.pi)            #reduced Planck constant
	Phi0    = 2.06783385 * 10**(-15) #flux quantum
	Carr    = Carr * 1e-15           #scale fF -> F
	L13     = L13 * 1e-12            #scale pH -> H
	L23     = L23 * 1e-12            #scale pH -> H
	phi_var = phiExt_fix[0]
	phi_fix = phiExt_fix[1]

	# Calculate juction energies in GHz
	Ej12 = Phi0/(2*np.pi) * Jc*wJ*lj12 * 1/h/1e9
	Ej22 = Phi0/(2*np.pi) * Jc*wJ*lj22 * 1/h/1e9
	Ej33 = Phi0/(2*np.pi) * Jc*wJ*lj33 * 1/h/1e9

	# Construct capacitance connectivity matrix
	Carr += np.array([0, Sc*wJ*lj12, 0, Sc*wJ*lj22, 0, Sc*wJ*lj33]) #!!1e15
	Cmat, Jmat = np.zeros((N,N)), np.zeros((N,N))
	Cmat[np.triu_indices(N,k=0)] = Carr
	Cmat = np.maximum(Cmat, Cmat.transpose())

	# Maxwell capacitance matrix C (not confused with Capacitance connectivity matrix Cmat)
	C = np.diag(np.sum(Cmat, axis=0)) + np.diag(np.diag(Cmat)) - Cmat
	Cinv = sp.linalg.inv(C)
	#print(Cinv*1e-15)

	# Rotation to decouple the oscillator and Josephson terms of H
	R = np.array([[1,0,1],[0,1,1],[1,1,1]])
	Rinv = np.array([[0, -1, 1], [-1, 0, 1], [1, 1, -1]])
	Cinv = Rinv.dot(Cinv.dot(Rinv))
	#print(Rinv.dot(R))


	# Define operators for node 1,2 (oscillator modes)
	I1     = qeye(n1)
	a1     = destroy(n1)
	adag1  = create(n1)
	adaga1 = num(n1)
	I2     = qeye(n2)
	a2     = destroy(n2)
	adag2  = create(n2)
	adaga2 = num(n2)

	# Define operators for node 3 (charge mode)
	I3 = I(n3)
	Q3 = Q(n3)
	Dp3 = Dp(n3)
	Dm3 = Dm(n3)

	### Oscillator Hamiltonian (in GHz) ###
	# Initialization of harmonic oscillator parameters and derived operators
	#-- mode 1 --
	f1     = np.sqrt(Cinv[0,0]/L23) / (2*np.pi) / 1.e9 #frequency of oscillator in GHz
	Z1     = np.sqrt(Cinv[0,0]*L23)                   #impedance of oscillator
	alpha1 = 2.*np.pi/Phi0 * np.sqrt(h*Z1/(4.*np.pi))   #displacement parameter
	Do1    = displace(n1, 1.j*alpha1)                  #quantum optical displacement operator
	#-- mode 2 --
	f2     = np.sqrt(Cinv[1,1]/L13) / (2.*np.pi) / 1.e9 #frequency of oscillator in GHz
	Z2     = np.sqrt(Cinv[1,1]*L13)                   #impedance of oscillator
	alpha2 = 2.*np.pi/Phi0 * np.sqrt(h*Z2/(4.*np.pi))   #displacement parameter
	Do2    = displace(n2, 1.j*alpha2)                  #quantum optical displacement operator
	# Assemble oscillator Hamiltonian
	Ho = f1 * tensor(adaga1, I2, I3) + f2 * tensor(I1, adaga2, I3)                     \
	     - Cinv[0,1]/(4.*np.pi*np.sqrt(Z1*Z2)) / 1.e9 * tensor(a1-adag1, a2-adag2, I3) \
	     - 1.0*Ej12/2. * ( np.exp(2.*np.pi*1.j*phi_var) * tensor(Do1.dag(), Do2.dag(), I3) + \
	     	          np.exp(-2.*np.pi*1.j*phi_var) * tensor(Do1, Do2, I3) )

	### Josephson Hamiltonian (in GHz) ###
	Hj = 1.0*Cinv[2,2]/2. * (2.*e)**2 / h / 1.e9 * tensor(I1, I2, Q3**2)

	### Interaction Hamiltonian ###
	Hint = -1.j * Cinv[0,2]/np.sqrt(4.*np.pi*Z1*h) / 1.e9 * (2.*e) * tensor(a1-adag1, I2, Q3) \
	       -1.j * Cinv[1,2]/np.sqrt(4.*np.pi*Z2*h) / 1.e9 * (2.*e) * tensor(I1, a2-adag2, Q3) \
	       -1.0*Ej22/2. * ( np.exp(-2.*np.pi*1.j*phi_fix) * tensor(I1, Do2.dag(), Dm3)        \
	       	                + np.exp(2.*np.pi*1.j*phi_fix) * tensor(I1, Do2, Dp3) )           \
	       -1.0*Ej33/2. * ( np.exp(0.) * tensor(Do1.dag(), Do2.dag(), Dm3)                    \
	       	                + np.exp(0.) * tensor(Do1, Do2, Dp3) )

	# Assemble full Hamiltonian
	H = 1.0*Ho + 1.0*Hj + 1.0*Hint

	# print('H.dims:', H.dims)
	# print('H.isherm:', H.isherm)
	print('*')

	evals, state = H.tidyup().eigenstates(eigvals=5, sparse=True)
	
	# # Visualize eigenstate state population
	# #print(abs(state[0].full().T[0]))
	# plt.figure()
	# for s in state:
	# 	plt.plot(abs(s.full().T[0]))
	# plt.show()

	return evals


if __name__=='__main__':

	#print(displace(5, 0.1))

	# Qubit parameters (Jc, Sc defined in function above)
	Carr = np.array([0., 0., 85.677, 4.455, 16.832, 70.556]) #in fF #np.array([0., 0., 98.847, 14.99, 38.666, 67.457])
	L13  = 289.395  #in pH #293.268
	L23  = 120.416  #in pH #98.07
	lj12 = 3.75498  #in um #3.7549
	lj22 = 0.395517 #in um #0.376366
	lj33 = 0.373288 #in um #0.386997

	# Truncation setting
	trunc = [12,12,12 ] #[6,6,8] #[12,12,12]

	# Single flux at operating point
	evals = spec_circuitC(Carr, L13, L23, lj12, lj22, lj33, trunc=trunc)
	print('Normalized eigen-energies at operating point (in GHz):', evals[0]-evals[0,0])

	# # Effect of truncation
	# xdata = np.arange(6,15,1)
	# ydata = []
	# for p in xdata:
	# 	trunc = [p,p,10]
	# 	evals = spec_circuitC(Carr, L13, L23, lj12, lj22, lj33, trunc=trunc)
	# 	ydata.append( evals[0,1]-evals[0,0] )
	# plt.figure()
	# plt.plot(xdata, ydata)
	# plt.show()

	# Spectrum
	phiVar = np.linspace(0.0-1*0.025, 0.0+1*0.025, 21, endpoint=True)
	# phiVar = np.linspace(0.0-0.5, 0.0+0.5, 21, endpoint=True)
	phiExt = [[p, 0.5] for p in phiVar]
	spec = spec_circuitC(Carr, L13, L23, lj12, lj22, lj33, phiExt=phiExt, trunc=trunc)

	print(spec[10,0]-np.min(spec))

	plt.figure()
	plt.plot(phiVar, spec[:,:2]-np.min(spec))
	plt.show()


















