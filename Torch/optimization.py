from torch.optim import Adam,LBFGS
import torch,pandas
from torch import tensor
from numpy import array
from scipy.optimize import minimize
from time import perf_counter
from DiSuQ.Torch.components import L,J,C
from DiSuQ.utils import empty
from DiSuQ.Torch.parallel import stackSpectrum,packSpectrum
import torch.distributed as dist

# PROFILE == MODULE
class Optimization:
    def __init__(self,circuit,profile,flux_profile=[],loss_function=None,distributor=None):
        self.circuit = circuit
        self.profile = profile # data parallel - control profile
        self.parameters,self.IDs = self.circuitParameters()
        self.Bounds = self.parameterBounds()
        self.flux_profile = flux_profile
        self.loss_function = loss_function
        self.distributor = distributor
        # depending upon the choice of optimization & loss
        self.vectors_calc = False
        self.grad_calc = True

    def logInit(self):
        self.logs = []
        self.dParams = [self.parameterState()]
        self.dCircuit = [self.circuitState()]

    def logCompile(self):
        dLog = pandas.DataFrame(self.logs)
        if len(dLog)>0:
            dLog['time'] = dLog['time'].diff()
        else:
            #import pdb;pdb.set_trace()
            print('Failure initial')
        dParams = pandas.DataFrame(self.dParams)
        dCircuit = pandas.DataFrame(self.dCircuit)
        return dLog,dParams,dCircuit

    def circuitID(self):
        IDs = []
        for component in self.circuit.network:
            IDs.append(component.ID)
        return IDs

    def circuitParameters(self,subspace=()):
        parameters = []; IDs = []
        for ID,parameter in self.circuit.named_parameters(subspace):
            parameters.append(parameter)
            IDs.append(ID)
        return parameters,IDs

    def circuitState(self):
        circuit = self.circuit
        parameters = {}
        for component in circuit.network:
            parameters[component.ID] = component.energy().item()
        return parameters

    def parameterState(self):
        circuit = self.circuit
        parameters = {}
        for component in circuit.network:
            if component.__class__ == C :
                parameters[component.ID] = component.cap.item()
            elif component.__class__ == L :
                parameters[component.ID] = component.ind.item()
            elif component.__class__ == J :
                parameters[component.ID] = component.jo.item()
        return parameters

    def parameterBounds(self):
        components = self.circuit.circuitComposition()
        Bounds = []
        for iD in components:
            bound = components[iD].bounds()
            Bounds.append(bound) # positive boundary exclusive
        return Bounds

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def spectrumProfile(self):
        Spectrum = self.profile(self.flux_profile)
        if self.distributor:
            # distribute over devices : Tensor stack packaging
            energy,state = stackSpectrum(Spectrum)
            Energy,State = self.distributor.placeholder()
            # gather Spectrum from all process
            dist.all_gather(Energy,energy)
            dist.all_gather(State,state)
            # replace Host data
            rank = dist.get_rank()
            Energy[rank] = energy
            State[rank] = state
            # flatten the data
            Spectrum = packSpectrum(Energy,State)
        return Spectrum

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def loss(self):
        Spectrum = self.spectrumProfile()
        loss,metrics = self.loss_function(Spectrum,self.flux_profile)
        return loss,metrics,Spectrum

    def backProp(self,loss):
        loss.backward(retain_graph=True)
        if self.distributor:
            for parameter in self.parameters:
                dist.all_reduce(parameter.grad.data,op=dist.ReduceOp.SUM)

    def lossScape(self,loss_function,flux_profile,scape,static):
        # parascape : {A:[...],B:[....]} | A,B in circuit.parameters
        A,B = scape.keys()
        Loss = empty((len(scape[A]),len(scape[B])))
        for id_A,a in enumerate(scape[A]):
            for id_B,b in enumerate(scape[B]):
                point = static.copy()
                point.update({A:a,B:b})
                self.circuit.initialization(point)
                loss,metrics,Spectrum = self.loss()
                Loss[id_A,id_B] = loss.detach().item()
        return Loss.transpose() # A -> X-axis , B -> Y-axis

    def breakPoint(self,logs):
        # break optimization loop : stagnation / spiking
        loss = pandas.DataFrame(logs)['loss'].to_numpy()
        if loss[-1] > 1e12 and len(loss) > 10:
            return True    
        return False

    def gradients(self):
        gradients = [parameter.grad.detach().item() for parameter in self.parameters]
        gradients = dict(zip(['grad-'+ID for ID in self.IDs],gradients))
        return gradients

    def multiInitialization(self,grid):
        return logs

class Minimization(Optimization):
    """
        * implementation : scipy minimization
        * gradient-free optimization methods
        * exception : LBFGS
    """

    def __init__(self,circuit,profile,flux_profile=[],loss_function=None,subspace=(),distributor=None):
        super().__init__(circuit,profile,flux_profile,loss_function,distributor)
        self.subspace = subspace
        self.static,self.keys,self.x0 = self.symmetryConstraint()

    def symmetryConstraint(self):
        x0 = self.circuitState()
        for slave,master in self.circuit.pairs.items():
            del x0[slave]
        static = dict() # static complement to subspace
        for key,value in x0.items():
            if not key in self.subspace and len(self.subspace)>0:
                static[key] = value
        for key in static:
            del x0[key]
        keys = tuple(x0.keys()) # subspace
        x0 =  array(tuple(x0.values())) # subspace
        return static,keys,x0

    def parameterInitialization(self,parameters):
        parameters = dict(zip(self.keys,parameters))
        parameters.update(self.static)
        for slave,master in self.circuit.pairs.items():
            parameters[slave] = parameters[master]
        self.circuit.initialization(parameters)
        self.parameters,self.IDs = self.circuitParameters(self.subspace) 

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def objective(self,parameters):
        self.parameterInitialization(parameters)
        if parameters is not self.parameters:
            # evaluate forward
            loss,metrics,Spectrum = self.loss()
        for parameter in self.parameters:
            if parameter.grad:
                parameter.grad.zero_()
        self.backProp(loss)
        parameters = dict(zip(self.IDs,self.parameters))
        gradients = []
        for iD in self.keys:
            parameter = parameters[iD]
            if parameter.grad is None: 
                # spectrum degeneracy:torch.eigh destablize
                # loss detached from computation graph
                gradients.append(0.0)
            else:
                gradients.append(parameter.grad.detach().item())
        return gradients

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def logger(self,parameters):
        self.parameterInitialization(parameters)
        loss,metrics,Spectrum = self.loss()
        loss = loss.detach().item()
        metrics['loss'] = loss
        metrics['time'] = perf_counter()-self.start # global scope
        self.logs.append(metrics)
        self.dParams.append(self.parameterState())
        self.dCircuit.append(self.circuitState())
        #torch.compiler.cudagraph_mark_step_begin()

    def optimization(self,method='Nelder-Mead',options=dict()):
        options['disp'] = True
        self.start = perf_counter()
        self.logInit()
        results = minimize(self.objective,self.x0,method=method,options=options,jac=self.gradients,bounds=self.Bounds,callback=self.logger)
        return self.logCompile()

class GradientDescent(Optimization):
    """
        * gradient descent optimization methods: Adam, LBFGS
        * torch optimization implementation
    """

    def __init__(self,circuit,profile,flux_profile=[],loss_function=None,distributor=None):
        super().__init__(circuit,profile,flux_profile,loss_function,distributor)
        self.log_grad = False
        self.log_spectrum = False
        self.log_hessian = False
        self.iteration = 0
        self.initAlgo()

    def initAlgo(self,algo=Adam,lr=1e-3):
        self.optimizer = algo(self.parameters,lr)

    def initLBFGS(self,lr=None,tol=1e-6,max_iter=20,history_size=5):
        line_search = None
        if lr is None:
            line_search = 'strong_wolfe'
        optimizer = LBFGS(self.parameters,lr=lr,max_iter=max_iter,history_size=history_size,tolerance_change=tol,
                              line_search_fn=line_search)
        return optimizer

    def logger(self,metrics,Spectrum):
        """
            * log every Spectrum decomposition
            * indexing via iteration count
            * additional gradient,Hessian logging
            * assuming apriori backward call
        """
        self.dParams.append(self.parameterState())
        self.dCircuit.append(self.circuitState())
        if self.log_spectrum:
            spectrum = dict()
            for level in range(3):
                spectrum['0-level-'+str(level)] = Spectrum[0][0][level].detach().item()
                spectrum['pi-level-'+str(level)] = Spectrum[int(len(flux_profile)/2)][0][level].detach().item()
            metrics.update(spectrum)
        if self.log_grad:
            gradients = self.gradients()
            metrics.update(gradients)
        #if self.log_hessian:
            #hess = hessian(self.lossModel(loss_function,flux_profile))
            #hess = tensor(hess)                    
            #hess = dict(zip(['hess'+index for index in range(len(hess))],hess))
            #metrics.update(hess)
        self.logs.append(metrics)

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def closure(self):
        """
            * erase the grad
            * reevaluates the model and returns the loss
            * calculate gradient
            * register the logs
            * in-place parameter update
        """
        self.optimizer.zero_grad()
        loss,metrics,Spectrum = self.loss()
        metrics['loss'] = loss.detach().item()
        metrics['iter'] = self.iteration
        self.backProp(loss)
        self.logger(metrics,Spectrum)
        return loss

    def optimization(self,iterations=100):
        start = perf_counter()
        self.logInit()
        for self.iteration in range(iterations):
            #torch.compiler.cudagraph_mark_step_begin()
            self.optimizer.step(self.closure)
            self.logs[-1]['time'] = perf_counter()-start
            if self.breakPoint(self.logs[-15:]):
                print('Optimization Break Point :',self.iteration)
                break

        return self.logCompile()

if __name__=='__main__':
    import torch
    from torch import tensor
    from DiSuQ.Torch.optimization import lossTransition
    from DiSuQ import utils
    from DiSuQ.Torch.models import transmon,fluxonium,zeroPi
    from datetime import timedelta
    torch.set_num_threads(36)
    cuda0 = torch.device('cuda:0')
    cpu = torch.device('cpu')
    torch.set_default_device(cpu)
    from models import transmon
    import torch.distributed as dist
    from torch.distributed import TCPStore
    #store = TCPStore('localhost',12345)
    #dist.init_process_group(backend=DISTRIBUTION_BACKEND,world_size=1,rank=0,store=store)
    from DiSuQ.Torch.components import DISTRIBUTION_BACKEND
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.nn.parallel import DataParallel as DP
    from circuit import Charge,Kerman
    basis = [1500]
    circuit = transmon(Charge,basis,sparse=False)
    cuda1 = torch.device('cuda:1')
    cpu = torch.device('cpu')
    torch.set_default_device(cuda0)
    from DiSuQ.Torch.circuit import Charge,Kerman
    #basis = [15,15,15]
    #flux_profile = Tensor([[0.],[.15],[.30],[.5]])
    #circuit = zeroPi(Charge,basis,sparse=False)
    flux_profile = [dict()]
    circuit = transmon(Charge,basis,sparse=False)
    flux_profile = tensor([[0.],[.15],[.30],[.5]],device=None)
    circuit = fluxonium(Charge,basis,sparse=False)
    from DiSuQ.Torch.optimization import GradientDescent
    loss = lossTransition(tensor(5.),tensor(4.5))
    optim = GradientDescent(circuit,circuit,flux_profile,loss)
    optim.optimizer = optim.initAlgo(lr=1e-2)
    print(circuit.circuitComponents())
    dLogs,dParams,dCircuit = optim.optimization(100)
    print(dLogs)
    optim = Minimization(circuit,circuit,flux_profile,loss)
    dLogs,dParams,dCircuit = optim.optimization()
    print(dLogs)
    dist.destroy_process_group()
    print("refer Optimization Verification")
