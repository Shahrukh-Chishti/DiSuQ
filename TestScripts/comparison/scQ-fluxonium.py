from DiSuQ.Torch import models
from torch import tensor,set_num_threads,stack
set_num_threads(32)
from numpy import array,linspace
import scqubits as scq
from DiSuQ.Torch.circuit import Oscillator
from DiSuQ.utils import plotCompare

Ej = 30.02 ; Ec = 10.2
El = .5
N = 256
flux_range = linspace(0,1,21)
limit = 4

def comparison(N):
    basis = [N]
    SCQ = []
    for flux in flux_range:
        fluxonium = scq.Fluxonium(EJ = Ej,
                                EC = Ec,
                                EL = El,flux=flux,
                                cutoff = N)
        spectrum = fluxonium.eigenvals(evals_count=limit)
        SCQ.append(spectrum)
    #SCQ = fluxonium.get_spectrum_vs_paramvals('flux',flux_range,limit)
    SCQ = array(SCQ).T
    circuit = models.fluxonium(Oscillator,basis,El,Ec,Ej,sparse=False)
    print('Fluxonium Parameters:',circuit.circuitComponents())
    flux_manifold = [[tensor(flux)] for flux in flux_range]
    DiS = circuit.spectrumManifold(flux_manifold)
    DiS = stack([val for val,vec in DiS]).detach().numpy().T
    return SCQ,DiS

if __name__ == '__main__':
    print('SCQubit Fluxonium Spectrum')
    SCQ,DiS = comparison(N)
    plotCompare(flux_range,
                {'SCQ-0':SCQ[0],'SCQ-1':SCQ[1],'SCQ-2':SCQ[2],
                 'DiSuQ-0':DiS[0],'DiSuQ-1':DiS[1],'DiSuQ-2':DiS[2]},
                 'Fluxonium Spectrum - Library comparison','flux_ext','spectrum(GHz)',width=5)