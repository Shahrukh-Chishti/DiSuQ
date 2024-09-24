from DiSuQ.Torch import models
from torch import tensor
from torch.linalg import eigvalsh
from torch import set_num_threads
set_num_threads(32)
from numpy import array,linspace
import scqubits as scq
from DiSuQ.Torch.circuit import Charge
from DiSuQ.utils import plotCompare

Ej = 30.02 ; Ec = 10.2
Ej = Ec*1.
N = 512
ng_list = linspace(-2, 2, 220)

def comparison(N):
    basis = [N]
    circuit = models.transmon(Charge,basis,Ej,Ec,sparse=False)
    H_LC = circuit.hamiltonianLC()
    H_J = circuit.hamiltonianJosephson()

    print('Transmon Parameters:',circuit.circuitComponents())

    SCQ,DiS = [],[]

    for q in ng_list:
        # calculating eigspectrum at each charge offset
        tmon = scq.Transmon(EJ=Ej,EC=Ec,ng=q,ncut=512)
        spectrum = tmon.eigenvals(evals_count=4)
        SCQ.append(spectrum)
        offset = dict([(1,tensor(q))])
        H_off = circuit.chargeChargeOffset(offset)
        H = H_LC+H_J+H_off
        spectrum = eigvalsh(H)[:4]
        DiS.append(spectrum.detach().numpy())

    return array(SCQ).T,array(DiS).T

if __name__ == '__main__':
    print('SCQubit Transmon Spectrum')
    tmon = scq.Transmon(
        EJ=Ej,
        EC=Ec,
        ng=0.0,
        ncut=N)

    #fig, axes = tmon.plot_evals_vs_paramvals('ng', ng_list, evals_count=3, subtract_ground=False)
    SCQ,DiS = comparison(N)
    plotCompare(ng_list,
                {'SCQ-0':SCQ[0],'SCQ-1':SCQ[1],'SCQ-2':SCQ[2],
                 'DiSuQ-0':DiS[0],'DiSuQ-1':DiS[1],'DiSuQ-2':DiS[2]},
                 'Transmon Spectrum - Library comparison','n_g','spectrum(GHz)',width=5)