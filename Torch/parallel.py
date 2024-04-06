from torch import stack,zeros
import torch,os
import torch.distributed as dist

def stackSpectrum(Spectrum):
    Energy,State = [],[]
    for energy,state in Spectrum:
        Energy.append(energy)
        State.append(state)
    energy,state = stack(Energy),stack(State)
    return energy,state

def packSpectrum(Energy,State):
    Spectrum = []
    for energy_,state_ in zip(Energy,State):
        # collection of control blocs
        for energy,state in zip(energy_,state_):
            Spectrum.append((energy,state))
    return Spectrum

class Distributor:
    def __init__(self,circuit,Rep,basis,control,world_size,splits=[],sparse=False,epochs=100,lr=1e-2):
        # Circuit Description
        self.circuit = circuit # circuit builder
        self.Rep = Rep # representation class
        self.sparse = sparse
        self.basis = basis

        # Control Description
        # list of control tensors
        # distributed over controlIDs
        self.control = control
        # if len(control)>0:
        #     assert len(control[0]) == len(control_iDs)

        # Optimization Description
        self.epochs = epochs
        self.lr = lr

        # Distribution
        if len(splits)==0:
            splits = [1]*len(self.control)
        self.splits = splits
        # size of the process group
        assert world_size == len(self.splits)
        self.world_size = world_size

    def circuitBuilder(self):
        circuit = self.circuit(self.Rep,self.basis,sparse=self.sparse)
        self.N = circuit.basisSize() # dimension of Hilbert Space
        self.levels = circuit.spectrum_limit
        # control_iDs = circuit.
        # self.control_iDs = control_iDs
        return circuit

    def distributeData(self,rank):
        index = sum(self.splits[:rank])
        # tensor partioning
        control = self.control[index:index+self.splits[rank]]
        control = control.to(rank)
        return control

    def placeholder(self):
        Energy,State = [],[]
        for split in self.splits:
            Energy.append(zeros(split,self.levels))
            State.append(zeros(split,self.N,self.levels))
        return Energy,State

    def distributionInit(self,rank):
        # global scope sub-routines
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.set_default_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=self.world_size)

def executionSetup(rank:int,distributor:Distributor):
    distributor.distributionInit(rank)
    from DiSuQ.Torch.optimization import GradientDescent,lossTransition
    flux_point = distributor.distributeData(rank)
    loss = lossTransition(torch.tensor(5.),torch.tensor(4.5))
    circuit = distributor.circuitBuilder()
    optim = GradientDescent(circuit,circuit,flux_point,loss,distributor)
    optim.optimizer = optim.initAlgo(lr=distributor.lr)
    dLogs,dParams,dCircuit = optim.optimization(distributor.epochs)
    print(dLogs)
    dist.destroy_process_group()

if __name__=='__main__':
    from DiSuQ.Torch.models import transmon,fluxonium,zeroPi
    from DiSuQ.Torch.circuit import Charge,Kerman
    world_size = torch.cuda.device_count()
    flux_profile = torch.tensor([[0.],[.15],[.30],[.5]],device=None)
    basis = [1500]
    splits = [2,2]
    distributor = Distributor(fluxonium,Charge,basis,flux_profile,world_size,splits)
    import torch.multiprocessing as mp
    mp.spawn(executionSetup,args=(distributor,),nprocs=world_size)

