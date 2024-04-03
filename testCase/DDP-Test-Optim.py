import sys,os,torch
import multiprocess as multi
# https://stackoverflow.com/questions/48846085/python-multiprocessing-within-jupyter-notebook
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import tensor
from DiSuQ.Torch.optimization import GradientDescent,lossTransition
from DiSuQ import utils
from DiSuQ.Torch.models import transmon,fluxonium,zeroPi
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from DiSuQ.Torch.circuit import Charge,Kerman

torch.set_num_threads(36)
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cpu = torch.device('cpu')
torch.set_default_device(cuda0)

world_size = torch.cuda.device_count()
# world_size = 1
#basis = [15,15,15]
#flux_profile = Tensor([[0.],[.15],[.30],[.5]])
#circuit = zeroPi(Charge,basis,sparse=False)
#basis = [1500]
#flux_profile = [dict()]
#circuit = transmon(Charge,basis,sparse=False)

# control data mapping to ranks
flux_profile = tensor([[0.],[.15],[.30],[.5]])
# flux_profile = tensor([[0.],[.5]])
flux_profile = [flux_profile[:2],flux_profile[2:]]

def circuitBuilder(rank):
    basis = [1500]
    circuit = fluxonium(Charge,basis,sparse=False)
    return circuit

def distributionSetup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #torch.cuda.device(rank)
    # indepdent process with independent default initialization
    torch.set_default_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def executionSetup(rank:int):
    print(rank,torch.cuda.current_device())
    distributionSetup(rank)
    # flux_point = flux_profile[[rank]]
    flux_point = flux_profile[rank]
    flux_point = flux_point.to(rank)
    circuit = circuitBuilder(rank)
    # for parameter in circuit.parameters():
    #     print(parameter.device)
    # print(rank,torch.cuda.current_device())
    loss = lossTransition(tensor(5.),tensor(4.5))
    print(rank,torch.cuda.current_device())
    module = DDP(circuit,device_ids=[rank])
    optim = GradientDescent(circuit,module,flux_point,loss)
    optim.optimizer = optim.initAlgo(lr=1e-2)
    dLogs,dParams,dCircuit = optim.optimization(100)
    print(dLogs)
    dist.destroy_process_group()
    # returns nothing since result sync hang
    # https://discuss.pytorch.org/t/exception-process-0-terminated-with-exit-code-1-error-when-using-torch-multiprocessing-spawn-to-parallelize-over-multiple-gpus/90636

if __name__=='__main__':
    # spawn is non-functionary on Jupyter pickle serializing
    mp.spawn(executionSetup,nprocs=world_size)
    sys.exit(0)
    with multi.Pool(processes=world_size) as pool:
        pool.map(executionSetup,range(world_size))
