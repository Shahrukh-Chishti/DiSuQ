from numpy import sqrt,arange,linspace
from torch import tensor,stack
from DiSuQ.Torch.models import oscillatorLC,transmon,fluxonium,zeroPi
from DiSuQ.Torch.circuit import Charge,Oscillator,Kerman,hamiltonianEnergy
from DiSuQ.Torch.components import indEnergy
from DiSuQ.utils import plotCompare

flux_range = linspace(0,1,21)
flux_manifold = [[tensor(flux)] for flux in flux_range]

def harmonicOscillator(Rep,basis):
    El = .120;Ec = .0054
    print('resonator frequency:',sqrt(8*El*Ec))
    print('potential energy bound:',.25/2/indEnergy(El))
    circuit = oscillatorLC(Rep,basis,El,Ec,sparse=False)
    circuit.spectrum_limit = 50
    spectrum,_ = circuit.eigenSpectrum(())
    return spectrum.detach().numpy()

def fluxoniumSpectrum(Rep,basis):
    Ej = 1.20;Ec = .054;El = 5.
    print('resonator frequency:',sqrt(8*Ej*Ec))
    print('potential energy bound:',Ej)
    circuit = fluxonium(Rep,basis,El=El,Ec=Ec,Ej=Ej,sparse=False)
    spectrum = circuit.spectrumManifold(flux_manifold)
    spectrum = stack([val for val,vec in spectrum]).detach().numpy().T
    return spectrum

def zeroPiSpectrum(Rep,basis):
    Ej = 1.20;Ec = .054;El = 5.
    print('resonator frequency:',sqrt(8*Ej*Ec))
    print('potential energy bound:',Ej)
    circuit = zeroPi(Rep,basis,El=El,Ec=Ec,Ej=Ej,sparse=False,symmetry=True)
    flux_manifold = [[tensor(flux),tensor(0.)] for flux in flux_range]
    spectrum = circuit.spectrumManifold(flux_manifold)
    spectrum = stack([val for val,vec in spectrum]).detach().numpy().T
    return spectrum

if __name__ == '__main__':
    print('Comparison of different Basis of Quantization')
    print('Harmonic Oscillator-----')
    spectrumQ = harmonicOscillator(Charge,[512])
    spectrumO = harmonicOscillator(Oscillator,[100])
    plotCompare(arange(50),
                {'charge':spectrumQ,'oscillator':spectrumO},
                 'Oscillator Spectrum - Basis Representation','level','spectrum(GHz)',width=5)
    
    print('Fluxonium-----')
    spectrumQ = fluxoniumSpectrum(Charge,[512])
    spectrumO = fluxoniumSpectrum(Oscillator,[1024])
    plotCompare(flux_range,
                {'charge-0':spectrumQ[0],'charge-1':spectrumQ[1],'charge-2':spectrumQ[2],
                 'oscillator-0':spectrumO[0],'oscillator-1':spectrumO[1],'oscillator-2':spectrumO[2]},
                 'Fluxonium Spectrum - Basis Representation','flux_ext','spectrum(GHz)',width=5)
    
    print('ZeroPi-----')
    spectrumQ = zeroPiSpectrum(Charge,[5,5,5])
    spectrumO = zeroPiSpectrum(Oscillator,[5,5,5])
    spectrumK = zeroPiSpectrum(Kerman,basis={'O':[10,10],'J':[5],'I':[]})
    plotCompare(flux_range,
                {'charge-0':spectrumQ[0],'charge-1':spectrumQ[1],'charge-2':spectrumQ[2],
                 'kerman-0':spectrumK[0],'kerman-1':spectrumK[1],'kerman-2':spectrumK[2],
                 'oscillator-0':spectrumO[0],'oscillator-1':spectrumO[1],'oscillator-2':spectrumO[2]},
                 'ZeroPi Spectrum - Basis Representation','flux_ext','spectrum(GHz)',width=5)
    