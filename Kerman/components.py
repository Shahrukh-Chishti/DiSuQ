
class Elements:
    def __init__(self,plus,minus):
        self.plus = plus
        self.minus = minus

class J(Elements):
    def __init__(self,plus,minus,energy):
        super(J, self).__init__(plus,minus)
        self.energy = energy

class C(Elements):
    def __init__(self,plus,minus,capacitance):
        super(J, self).__init__(plus,minus)
        self.capacitance = capacitance

class L(Elements):
    def __init__(self,plus,minus,inductance):
        super(J, self).__init__(plus,minus)
        self.inductance = inductance
