from DiSuQ.Torch.models import shuntedQubit
from DiSuQ.Torch.circuit import Kerman
from DiSuQ.utils import plotCompare
from torch import tensor,stack
from numpy import linspace
from time import perf_counter
import warnings
warnings.filterwarnings('ignore')

flux_range = tensor(linspace(0,1,4))
flux_manifold = [[flux] for flux in flux_range]
flux_point = ('I')

if __name__ == '__main__':
    print('Comparing Resource Allocation and Accuracy of Sparse/Dense backend')
    print('Shunted Flux Qubit-------')
    basis = {'O':[15],'J':[8,8],'I':[]}

    circuit = shuntedQubit(Kerman,basis,sparse=False)
    circuit.grad_calc = False
    print(circuit.circuitComponents())
    start = perf_counter()
    spectrum = circuit.spectrumManifold(flux_manifold)
    print('Time(dense):',perf_counter()-start)
    dense = stack([val for val,vec in spectrum]).detach().numpy().T

    circuit = shuntedQubit(Kerman,basis,sparse=True)
    circuit.grad_calc = False
    print(circuit.circuitComponents())
    start = perf_counter()
    spectrum = circuit.spectrumManifold(flux_manifold)
    print('Time(sparse):',perf_counter()-start)
    sparse = stack([val for val,vec in spectrum]).detach().numpy().T

    plotCompare(flux_range,
                {'dense-0':dense[0],'dense-1':dense[1],'dense-2':dense[2],
                 'sparse-0':sparse[0],'sparse-1':sparse[1],'sparse-2':sparse[2]},
                 'Shunted Flux Spectrum - Backend comparison','flux_ext','spectrum(GHz)',width=5)








