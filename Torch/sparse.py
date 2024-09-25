from numpy import abs as absolute
from torch import kron as kronecker
from torch import vstack,transpose
from torch import matmul as mul,nonzero
from torch import sparse,sparse_coo_tensor
from torch.sparse import mm as mul
from scipy import sparse

from .components import *

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

def modeProduct(A,i,B,j):
    return mul(basisProduct(A,[i]),basisProduct(B,[j]))

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
            if not M[i,j]==0:
                H += M[i,j]*modeProduct(A,i+a,B,j+b)
    return H

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

def unitaryTransformation(M,U):
    M = mul( transpose(U.conj(),0,1), mul(M, U))
    return M

""" Sparse Methods """

def indicesSparse(A):
    return A.coalesce().indices().T

def valuesSparse(A):
    return A.coalesce().values()

def scipyfy(H):
    indices = indicesSparse(H).numpy().T
    values = valuesSparse(H).detach().numpy()
    shape = H.size()
    return sparse.coo_matrix((values,indices),shape=shape)

def sparsify(T,dense=None,device=None):
    indices = nonzero(T,as_tuple=True)
    shape = T.shape
    values = T[indices]
    indices = vstack(indices)
    return sparse_coo_tensor(indices,values,shape,device=device) ##

def diagSparse(values,diagonal=0,device=None):
    N = len(values)+absolute(diagonal)
    rows = arange(N,dtype=int,device=device) ##
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
    return sparse_coo_tensor(indices,values,(N,N),device=device) ##

"""" Operator Tensors """

def identity(n,dtype=float,device=None):
    return sparsify(eye(n,dtype=dtype),device=device)

def null(shape=1,dtype=complex,device=None):
    return sparse_coo_tensor([[],[]],[],[shape]*2,dtype=dtype,device=device) ##

# States Grid

def chargeStates(n,dtype=int,device=None):
    charge = linspace(n,-n,2*n+1,dtype=dtype,device=device)
    return charge

def fluxStates(N_flux,n_flux=1,dtype=complex,device=None):
    flux = linspace(n_flux,-n_flux,N_flux+1,dtype=dtype,device=device)[1:]
    return flux

def transformationMatrix(n_charge,n_flux=1,device=None):
    charge_states = chargeStates(n_charge,device)
    N_flux = 2*n_charge+1 # dimensionality of Hilbert space
    flux_states = fluxStates(N_flux,n_flux,complex,device)/2/n_flux
    # domain of flux bound to (-.5,.5] : Fourier transform
    T = outer(flux_states,charge_states)
    T *= 2*pi*im # Fourier Phase
    T = exp(T)/N_flux # Normalization
    return T # unitary transformation

# Oscillator Basis

def basisQo(n,impedance,device=None):
    Qo = arange(1,n,device=device) ##
    Qo = sqrt(Qo)
    Qo = -diagSparse(Qo,1,device) + diagSparse(Qo,-1,device)
    return Qo*im*sqrt(1/2/pi/impedance)

def basisFo(n,impedance,device=None):
    Po = arange(1,n,device=device) ##
    Po = sqrt(Po)
    Po = diagSparse(Po,1,device) + diagSparse(Po,-1,device)
    return Po*sqrt(impedance/2/pi)

# Canonical Basis

def basisQq(n,device=None):
    # charge basis
    charge = chargeStates(n,device)
    Q = diagSparse(charge.clone().detach(),device=device) ##
    return Q * 2

def basisFqKerman(n):
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

def basisFq(n,device=None):
    Q = basisQq(n,device)
    U = transformationMatrix(n,device=device)
    return U@Q@U.conj().T/2

def basisFf(N,n=1,device=None):
    flux = fluxStates(N,n,device=device)/2/n
    F = diagSparse(flux)
    return F

def basisFf(N,n,device=None):
    flux = fluxStates(N,n,device=device)/2/n # periodicity bound
    F = diagSparse(flux)
    return F

def basisQf(n):
    F = basisFf(n).to(complex)
    U = transformationMatrix(n,2*n+1,n)
    return U@F@U.conj().T*(2*n+1)*2

# Derivative Stencil

def basisFiniteI(n,bound):
    delta = (bound[1]-bound[0])/(n-1)
    stencil = [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60]
    I = null(n)
    for index,coeff in enumerate(stencil):
        index -= 3
        I += diagSparse(tensor([coeff]*(n-numpy.abs(index))),index)
    return I/delta

def basisFiniteII(n,bound):
    delta = (bound[1]-bound[0])/(n-1)
    stencil = [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90]
    II = null(n)
    for index,coeff in enumerate(stencil):
        index -= 3
        II += diagSparse(tensor([coeff]*(n-numpy.abs(index))),index)
    return II/delta/delta    

# Junction Displacement Operators    

def chargeDisplacePlus(n,device=None):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex,device=device) ##
    D = diagSparse(diagonal,diagonal=-1)
    return D

def chargeDisplaceMinus(n,device=None):
    """n : charge basis truncation"""
    diagonal = ones((2*n+1)-1,dtype=complex,device=device) ## 
    D = diagSparse(diagonal,diagonal=1)
    return D

# better implementation of matrix exponential for sparse matrices

def displacementCharge(n,a,device=None):
    D = basisFq(n,device).to_dense()
    D = expm(im*2*pi*a*D)
    return sparsify(D,device=device)

def displacementOscillator(n,z,a,device=None):
    D = basisFo(n,z,device).to_dense()
    D = expm(im*2*pi*a*D)
    return sparsify(D,device=device)

def displacementFlux(n,a):
    flux = fluxStates(2*n+1,n)
    flux = exp(im*2*pi*a*flux)
    return diagSparse(flux)

def displacementFlux(N,n,a):
    flux = fluxStates(N,n)/2/pi
    flux = exp(im*2*pi*a*flux)
    return diagSparse(flux)

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
