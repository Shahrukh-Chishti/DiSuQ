import torch,numpy

def polynomial(poly,x):
    N = len(poly)-1
    func = torch.tensor(0.0)
    for index,coeff in enumerate(poly):
        power = N-index
        func += coeff*(x**(power))
    return func

def gradient(poly,x):
    N = len(poly)-1
    func = torch.tensor(0.0)
    for index,coeff in enumerate(poly[:-1]):
        power = N-index
        func += power*coeff*(x**(power-1))
    return func

def curvature(poly,x):
    N = len(poly)-1
    func = torch.tensor(0.0)
    for index,coeff in enumerate(poly[:-2]):
        power = N-index
        func += power*(power-1)*coeff*(x**(power-2))
    return func

def superNewtonUpdate(poly,x):
    f,I,II = polynomial(poly,x),gradient(poly,x),curvature(poly,x)
    return 1/(I/f-II/I)/2

def groundState(poly,x,iterations=10):
    # assuming x0 < ground state
    X = []
    for iteration in range(iterations):
        x = x-superNewtonUpdate(poly,x)
        X.append(x)
    return x,X

def squarePolynomial(poly):
    N = len(poly)
    poly = torch.cat(poly).view(1,N)
    square = poly.T@poly
    square = torch.fliplr(square)
    return [square.diagonal(offset=-k).sum() for k in numpy.arange(-N-1,N,1)]

def rootMinima(poly):
    poly = numpy.polynomial.Polynomial(numpy.flip(poly))
    roots = poly.roots()
    return min(roots)

def constructionHermitian(position,params,N):
    H = torch.zeros(N,N,dtype=torch.double)
    for index,pos in enumerate(position):
        x,y = pos.tolist()
        H[x,y] = params[index]
        H[y,x] = params[index]
    return H

def characteristicPolynomial(H):
    # Faddeev - LeVerrier algorithm
    N = len(H)
    poly = [torch.tensor([1.],dtype=torch.double)]
    M = torch.zeros(N,N,dtype=torch.double)
    for k in range(N):
        M = H@M + poly[-1]*torch.eye(N,dtype=torch.double)
        poly.append(-torch.trace(H@M).view(1)/(k+1))
    return poly