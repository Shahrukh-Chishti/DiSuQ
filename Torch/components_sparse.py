from numpy import abs as absolute

from torch import kron as kronecker
from torch import vstack,transpose
from torch import matmul as mul,nonzero
from torch import sparse,sparse_coo_tensor
from torch.sparse import mm as mul

from components import *

def unitaryTransformation(M,U):
    M = mul( transpose(U.conj(),0,1), mul(M, U))
    return M

def indicesSparse(A):
    return A.coalesce().indices().T

def valuesSparse(A):
    return A.coalesce().values()

def kron(A,B):
    nA,nB = A.shape[0],B.shape[0]
    N = nA*nB
    values = kronecker(valuesSparse(A),valuesSparse(B))
    idA = indicesSparse(A)*nB
    idB = indicesSparse(B)
    rows = (idA[:,0][:,None] + idB[:,0][None,:]).ravel()
    cols = (idA[:,1][:,None] + idB[:,1][None,:]).ravel()
    indices = vstack((rows,cols))
    return sparse_coo_tensor(indices,values,(N,N))

def sparsify(T):
    indices = nonzero(T,as_tuple=True)
    shape = T.shape
    values = T[indices]
    indices = vstack(indices)
    return sparse_coo_tensor(indices,values,shape)

def identity(n):
    return sparsify(eye(n))

def null(shape=1,dtype=complex64):
    return sparse_coo_tensor([[],[]],[],[shape]*2,dtype=dtype)

def diagSparse(values,diagonal=0):
    N = len(values)+absolute(diagonal)
    rows = arange(N,dtype=int)
    cols = rows.clone()
    if diagonal > 0:
        cols += diagonal
        cols = cols[:-diagonal]
        rows = rows[:-diagonal]
    elif diagonal < 0:
        rows -= diagonal
        rows = rows[:diagonal]
        cols = cols[:diagonal]
    indices = vstack([rows,cols])
    return sparse_coo_tensor(indices,values,(N,N))

def basisQo(n,impedance):
    Qo = arange(1,n)
    Qo = sqrt(Qo)
    Qo = -diagSparse(Qo,diagonal=1) + diagSparse(Qo,diagonal=-1)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance):
    Po = arange(1,n)
    Po = sqrt(Po)
    Po = diagSparse(Po,diagonal=1) + diagSparse(Po,diagonal=-1)
    return Po*sqrt(impedance/2/pi)

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
    return sparsify(T)

def basisQq(n):
    # charge basis
    charge = chargeStates(n).to(complex64)
    Q = diagSparse(charge.clone().detach())
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
    return unitaryTransformation(Q,transpose(U.conj(),0,1))/(2.0*n+1.0)/2.0

def basisFf(n):
    flux = fluxStates(2*n+1,n)
    F = diagSparse(flux)
    return F

def basisQf(n):
    F = basisFf(n).to(complex64)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex64)
    D = diagSparse(diagonal,diagonal=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex64)
    D = diagSparse(diagonal,diagonal=1)
    return D

# better implementation of matrix exponential for sparse matrices

def displacementCharge(n,a):
    D = basisFq(n).to_dense()
    D = expm(im*2*pi*a*D)
    return sparsify(D)

def displacementOscillator(n,z,a):
    D = basisFo(n,z).to_dense()
    D = expm(im*2*pi*a*D)
    return sparsify(D)

def displacementFlux(n,a):
    D = basisFf(n).to_dense()
    D = expm(im*2*pi*a*D)
    return sparsify(D)

if __name__=='__main__':
    Qo = basisQo(30,tensor(4.0))
    Fq = basisFq(30)
    Qf = basisQf(30)
    S1 = sparse_coo_tensor([[0],[0]],[1.0],[3,3])
    S2 = sparse_coo_tensor([[0],[0]],[4.0],[3,3])
    s1 = zeros(3,3);s1[0,0]=1;s1[1,2]=234
    s2 = zeros(3,3);s2[0,0]=4
    Sp1 = sparsify(s1)
    Sp2 = sparsify(s2)
    import ipdb;ipdb.set_trace()
