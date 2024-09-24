from DiSuQ.Torch.models import oscillatorLC,transmon,fluxonium,zeroPi
from DiSuQ.Torch.circuit import Kerman,Oscillator,Flux,Charge
from DiSuQ.Torch.components import complex,float
import warnings
warnings.filterwarnings('ignore')

def comparison(H,limit=4):
    from torch.linalg import eigvalsh,eigh
    print('torch.linalg.eigvalsh:',eigvalsh(H)[:limit])
    print('torch.linalg.eigh:',eigh(H)[0][:limit])
    from numpy.linalg import eigvalsh,eigh
    H = H.detach().numpy()
    print('numpy.linalg.eigvalsh:',eigvalsh(H)[:limit])
    print('numpy.linalg.eigh:',eigh(H)[0][:limit])

def main(N=1):
    print('Comparison of different Eigvalue problem solving algorithms, \n available from various libraries & modules')
    
    print('Harmonic Oscillator(q basis)-------')
    circuit = oscillatorLC(Charge,[N*128])
    H = circuit.hamiltonianLC()
    comparison(H)

    print('Transmon(q basis)-------')
    circuit = transmon(Charge,[N*128])
    H = circuit.hamiltonianLC() + circuit.hamiltonianJosephson()
    from torch import lobpcg
    print('torch.lobpcg:',lobpcg(H.to(float),k=4,largest=False,method='ortho')[0])
    comparison(H)

    print('Fluxonium(o basis)-------')
    circuit = fluxonium(Oscillator,[N*128])
    H = circuit.hamiltonianLC() + circuit.hamiltonianJosephson()
    comparison(H)

    print('Zero-Pi(q basis)-------')
    basis = {'O':[5,5],'I':[],'J':[5]}
    circuit = zeroPi(Kerman,basis)
    H = circuit.hamiltonianLC() + circuit.hamiltonianJosephson()
    comparison(H)

if __name__ == "__main__":
    main()