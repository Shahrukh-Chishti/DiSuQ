import numpy

def basisQo(n,impedance):
    return Qo*iota*numpy.sqrt(h/2/impedance)

def basisQji(n):
    return Qji*2*e

def basisPo(n,impedance):
    return Po*numpy.sqrt(h*impedance/2)

def basisPj(n):
    return Pj

def chargeDisplacePlus(n):
    return D

def chargeDisplaceMinus(n):
    return D

class Elements:
    def __init__(self,plus,minus):
        self.plus = plus
        self.minus = minus

class J(Elements):
    def __init__(self,plus,minus,energy):
        super().__init__(plus,minus)
        self.energy = energy

class C(Elements):
    def __init__(self,plus,minus,capacitance):
        super().__init__(plus,minus)
        self.capacitance = capacitance

class L(Elements):
    def __init__(self,plus,minus,inductance):
        super().__init__(plus,minus)
        self.inductance = inductance
