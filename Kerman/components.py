import numpy,uuid
from numpy import cos,sin,pi,exp

im = 1.0j
e = 1.60217662 * 10**(-19)
h = 6.62607004 * 10**(-34)
flux_quanta = h/2/e

def identity(n):
    return numpy.eye(n)

def basisQo(n,impedance):
    return Qo*iota*numpy.sqrt(h/2/impedance)

def basisQji(n):
    charge = numpy.linspace(n,-n,2*n+1,dtype=int)
    Qji = numpy.zeros((len(charge),len(charge)),int)
    numpy.fill_diagonal(Qji,charge)
    return Qji*2*e

def basisPo(n,impedance):
    return Po*numpy.sqrt(h*impedance/2)

def basisPj(n):
    N = 2*n+1
    P = numpy.zeros((N,N))
    charge = numpy.linspace(n,-n,N,dtype=int)
    for q in charge:
        for p in charge:
            if not p==q:
                P[q,p] = flux_quanta*((n+1)*sin(2*pi*(q-p)*n/N) - n*sin(2*pi*(q-p)*(n+1)/N))
                P[q,p] /= N*(1-cos(2*pi*(q-p)/N))*N
    return P

def chargeDisplacePlus(n):
    """n : charge basis truncation"""
    diagonal = numpy.ones((2*n+1)-1,dtype=numpy.complex)
    D = numpy.diag(diagonal,k=-1)
    return D

def chargeDisplaceMinus(n):
    """n : charge basis truncation"""
    diagonal = numpy.ones((2*n+1)-1,dtype=numpy.complex)
    D = numpy.diag(diagonal,k=1)
    return D

class Elements:
    def __init__(self,plus,minus,ID=None):
        self.plus = plus
        self.minus = minus
        if ID is None:
            ID = uuid.uuid4().hex
        self.ID = ID

class J(Elements):
    def __init__(self,plus,minus,energy,ID=None):
        super().__init__(plus,minus,ID)
        self.energy = energy * 10.**9

class C(Elements):
    def __init__(self,plus,minus,capacitance,ID=None):
        super().__init__(plus,minus,ID)
        self.capacitance = capacitance * 10.**(-15)

class L(Elements):
    def __init__(self,plus,minus,inductance,ID=None,external=False):
        super().__init__(plus,minus,ID)
        self.inductance = inductance * 10.**(-12)
        self.external = external


if __name__=='__main__':
    Q = basisQji(4)
    P = basisPj(4)
    import ipdb;ipdb.set_trace()
