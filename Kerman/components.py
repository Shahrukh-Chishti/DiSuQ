import numpy

def basisQo(modes):
    return basis

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
