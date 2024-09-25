import sys,os,torch
import multiprocess as multi
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import tensor

torch.set_num_threads(36)
cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cpu = torch.device('cpu')
torch.set_default_device(cuda0)

world_size = torch.cuda.device_count()

# Does distributed maintain autograd on host tensor ??
# It surely halts the autograd engine from independent processes

def distributionSetup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    #torch.cuda.device(rank)
    # indepdent process with independent default initialization
    torch.set_default_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def executionSetup(rank:int):
    distributionSetup(rank)
    # All tensors below are of torch.int64 dtype.
    # We have 2 process groups, 2 ranks.
    torch.set_default_device(rank)
    tensor_list = [torch.zeros(2, dtype=torch.float) for _ in range(2)]
    print(rank,tensor_list)
    leaf = torch.arange(2, dtype=torch.float, requires_grad=True)
    tensor = leaf*(1+rank) + 1 + 2 * rank
    print(rank,tensor)
    dist.all_gather(tensor_list, tensor)
    print(rank,tensor_list)
    print(rank, tensor_list[rank]==tensor)
    print(rank,tensor_list[rank].requires_grad)
    print(rank,tensor.requires_grad)

    # False :: recieved tensors are grad-free
    # even the host

    # inplace replacement
    tensor_list[rank] = tensor
    loss = tensor_list[0]+tensor_list[1]
    loss = loss.sum()
    loss.backward()
    #print(rank,loss.grad)
    print(rank,leaf.grad)

    # Identical Loss calculation

    dist.all_reduce(leaf.grad.data, op=dist.ReduceOp.SUM)
    print(rank,leaf.grad)
    dist.destroy_process_group()

if __name__=='__main__':
    print('Testing the base Distributed multiprocessing')
    mp.spawn(executionSetup,nprocs=world_size)
    print('pass on world size:',world_size)
    print('Distributed class is built over this !!')
    sys.exit(0)
