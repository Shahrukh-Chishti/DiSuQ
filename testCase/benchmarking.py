import qucat_circuits as qC
import kerman_circuits as kC
import utils,numpy

def transmon(fluxes):
    q_spectrum,k_spectrum = [],[]
    q_transmon = qC.transmon()
    k_transmon = kC.transmon([7])

    for flux in fluxes:
        H = k_transmon.hamiltonian_charged([flux])
        energy = numpy.linalg.eigvals(H)
        k_spectrum.append(min(energy).real)

        H = q_transmon.hamiltonian(Lj = qC.Lj(flux),excitations = 7,taylor = 4)
        q_spectrum.append(H.eigenenergies()[0])

    eigenspectrum = {'qucat':q_spectrum,'kerman':k_spectrum}
    utils.plotCompare(fluxes,eigenspectrum)
    return transmon

if __name__=='__main__':
    fluxes = numpy.linspace(-1,1,100)
    transmon(fluxes)
