from torch import stack,zeros
import torch,os
import torch.distributed as dist
from DiSuQ.Torch.parallel import Distributor

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


