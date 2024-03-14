from torch.optim import SGD,RMSprop,Adam,LBFGS,Rprop
import torch,pandas
from torch import tensor,Tensor,argsort,zeros,abs,mean,stack,var,log,isnan,lobpcg,std
from torch.autograd import grad
from torch.linalg import det,inv,eigh as eigsolve, eigvalsh
from torch.nn.utils import clip_grad_norm_,clip_grad_value_
from numpy import arange,set_printoptions,meshgrid,linspace,array,sqrt,sort,log10,random,logspace
from scipy.optimize import minimize
from time import perf_counter,sleep
from DiSuQ.Torch.components import L,J,C,L0,J0,C0
from DiSuQ.Torch.components import COMPILER_BACKEND
from DiSuQ.utils import empty
from scipy.stats import truncnorm

class Optimization:
    def __init__(self,circuit,profile,flux_profile=[],loss_function=None):
        self.circuit = circuit
        self.profile = profile # data parallel - control profile
        self.parameters,self.IDs = self.circuitParameters()
        self.Bounds = self.parameterBounds()
        self.flux_profile = flux_profile
        self.loss_function = loss_function
        # depending upon the choice of optimization & loss
        self.vectors_calc = False
        self.grad_calc = True
        self.logInit()

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
        slaves = self.circuit.pairs.keys()
        for component in self.circuit.network:
            if component.ID in subspace or len(subspace)==0:
                if component.__class__ == C :
                    parameter = component.cap
                elif component.__class__ == L :
                    parameter = component.ind
                elif component.__class__ == J :
                    parameter = component.jo
                if not component.ID in slaves:
                    IDs.append(component.ID)
                    parameters.append(parameter)
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
        # distribute over devices !!!
        # concat over DataParallel
        Spectrum = self.profile(self.flux_profile)
        return Spectrum
    
    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def loss(self):
        Spectrum = self.spectrumProfile()
        loss,metrics = self.loss_function(Spectrum,self.flux_profile)
        return loss,metrics,Spectrum
    
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

    def __init__(self,circuit,profile,flux_profile=[],loss_function=None,subspace=()):
        super().__init__(circuit,profile,flux_profile,loss_function)
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
        loss,metrics,Spectrum = self.loss()
        loss = loss.detach().item()
        return loss

    #@torch.compile(backend=COMPILER_BACKEND, fullgraph=False)
    def gradients(self,parameters):
        # method overriding super-class
        self.parameterInitialization(parameters)
        if parameters is not self.parameters:
            # evaluate forward
            loss,metrics,Spectrum = self.loss()
        for parameter in self.parameters:
            if parameter.grad:
                parameter.grad.zero_()
        loss.backward(retain_graph=False)
        parameters = dict(zip(self.IDs,self.parameters))
        gradients = []
        for iD in self.keys:
            parameter = parameters[iD]
            if parameter.grad is None: 
                # spectrum degeneracy:torch.eigh destablize
                # loss detached from computation graph
                #import ipdb;ipdb.set_trace()
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
        results = minimize(self.objective,self.x0,method=method,options=options,jac=self.gradients,bounds=self.Bounds,callback=self.logger)
        return self.logCompile()

class GradientDescent(Optimization):
    """
        * gradient descent optimization methods: Adam, LBFGS
        * torch optimization implementation
    """

    def __init__(self,circuit,profile,flux_profile=[],loss_function=None):
        super().__init__(circuit,profile,flux_profile,loss_function)
        self.log_grad = False
        self.log_spectrum = False
        self.log_hessian = False
        self.iteration = 0
        self.optimizer = self.initAlgo()

    def initAlgo(self,algo=Adam,lr=1e-3):
        optimizer = algo(self.parameters,lr)
        return optimizer

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
            * reevaluates the model and returns the loss
            * calculate gradient
            * register the logs
            * in-place parameter update
        """
        self.optimizer.zero_grad()
        loss,metrics,Spectrum = self.loss()
        metrics['loss'] = loss.detach().item()
        metrics['iter'] = self.iteration
        loss.backward(retain_graph=True)
        self.logger(metrics,Spectrum)
        return loss

    def optimization(self,iterations=100):
        start = perf_counter()
        for self.iteration in range(iterations):
            #torch.compiler.cudagraph_mark_step_begin()
            self.optimizer.step(self.closure)
            self.logs[-1]['time'] = perf_counter()-start
            if self.breakPoint(self.logs[-15:]):
                print('Optimization Break Point :',self.iteration)
                break

        return self.logCompile()

### GRID initialization

def truncNormalParameters(circuit,subspace,N,var=5):
    # var : std of normal distribution
    iDs,domain = [],[]
    for component in circuit.network:
        if component.ID in subspace:
            iDs.append(component.ID)
            loc = component.energy().item()
            a,b = component.bounds()
            a = (a - loc)/var ; b = (b - loc)/var
            domain.append(truncnorm.rvs(a,b,loc,var,size=N,random_state=random.seed(101)))
    grid = array(domain)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(iDs,point)))
        space.append(state)
    return space

def uniformParameters(circuit,subspace,N):
    iDs,domain = [],[]
    for component in circuit.network:
        if component.ID in subspace:
            iDs.append(component.ID)
            a,b = component.bounds()
            a = log10(a.item()); b = log10(b.item())
            domain.append(logspace(a,b,N+1,endpoint=False)[1:])
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(iDs),-1)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(iDs,point)))
        space.append(state)
    return space

def domainParameters(domain,circuit,subspace):
    grid = array(meshgrid(*domain))
    grid = grid.reshape(len(subspace),-1)
    space = []
    for point in grid.T:
        state = circuit.circuitState()
        state.update(dict(zip(subspace,point)))
        space.append(state)
    return space

def initializationSequential(parameters,optimizer,lossFunction,flux_profile,iterations=100,lr=.005):
    Search = []
    for index,parameter in enumerate(parameters):
        print(index,'--------------------')
        optimizer.circuit.initialization(parameter)
        optimizer.parameters,_ = optimizer.circuitParameters()
        Search.append(optimizer.optimization(lossFunction,flux_profile,iterations=iterations,lr=lr))
    return Search
    
def initializationParallelism(optimizer,lossFunction,flux_profile,iterations=100,lr=.005):
    def optimization(parameters):
        optimizer.circuit.initialization(parameters)
        optim.parameters,_ = optim.circuitParameters()
        return optimizer.optimization(lossFunction,flux_profile,iterations=iterations,lr=lr)
    return optimization
    
def anHarmonicityProfile(optim,flux_profile):
    anHarmonicity_profile = []
    for flux in flux_profile:
        spectrum,state = optim.spectrumOrdered(flux)
        anHarmonicity_profile.append(anHarmonicity(spectrum).item())
    return anHarmonicity_profile

### LOSS functions

MSE = torch.nn.MSELoss()

def anHarmonicity(spectrum):
    ground,Ist,IInd = spectrum[:3]
    return (IInd-Ist)-(Ist-ground)

# Full batch of flux_profile
def lossTransitionFlatness(Spectrum,flux_profile):
    spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
    loss = var(spectrum[:,1]-spectrum[:,0])
    loss += var(spectrum[:,2]-spectrum[:,1])
    return loss,dict()
        
# def std(E):
#     N = len(E)
#     profile = tensor([0.]*N)
#     for index,e in enumerate(E):
#         profile += e

def lossDegeneracyWeighted(delta0,D0,N=2):
    def Loss(Spectrum,flux_profile):
        spot = 0
        
        E10 = [E[0][1]-E[0][0] for E in Spectrum]
        E20 = [E[0][2]-E[0][0] for E in Spectrum]
        
        Ist = abs(E10[0]-E10[1])
        #E10 = tensor(E10,requires_grad=True)

        # n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        #        degen_point = flux_profile[0]['Lx']
        #delta = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]-e10
        #delta = delta/e10
        #delta = delta.abs()
        #sensitivity = grad(e10,degen_point,create_graph=True)[0] # local fluctuations
        #sensitivity = log(sensitivity.abs())
        
        D = log((E20[0])/(E10[0]))/log(tensor(10.))
        delta = log(Ist/E10[0])/log(tensor(10.))
        loss = delta*delta0 - D*D0
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':E10[0].detach().item(),'E20':E20[0].detach().item()}
    return Loss

def lossDegeneracyTarget(delta0,D0):
    def Loss(Spectrum,flux_profile):
        half = 0#int(len(flux_profile)/2)
        neighbour = -1
        e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
        e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
        D = log(e20/e10)
        degen_point = flux_profile[0]['Lx']
        n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
        delta = log((n10-e10).abs()/e20)
        loss = (delta0+delta)**2 + (D0-D)**2
        return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':e10.detach().item(),'E20':e20.detach().item()}
    return Loss

# def lossDegeneracy(Spectrum,flux_profile):
#     half = 0#int(len(flux_profile)/2)
#     neighbour = -1
#     e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
#     e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
#     D = log(e20/e10)
#     degen_point = flux_profile[0]['Lx']
#     n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
#     delta = log((n10-e10).abs())
#     #import pdb;pdb.set_trace()
#     a = 1/(log(n10).abs()).detach().item() ; b = 1/(log(e10).abs()).detach().item()
#     k = sqrt(2/(a**2+b**2))
#     loss = k*(a*log(n10) + b*log(e10))
#     return loss,{'delta':delta.detach().item(),'D':D.detach().item(),'E10':e10.detach().item()}


# def lossDegeneracy(Spectrum,flux_profile):
#     half = 0#int(len(flux_profile)/2)
#     neighbour = -1
#     e20 = Spectrum[half][0][2]-Spectrum[half][0][0]
#     e10 = Spectrum[half][0][1]-Spectrum[half][0][0]
#     D = log(e20/e10)
#     degen_point = flux_profile[0]['Lx']
#     n10 = Spectrum[neighbour][0][1]-Spectrum[neighbour][0][0]
#     loss = log(n10)
#     return loss,{'D':D.detach().item(),'N10':n10.detach().item(),'E10':e10.detach().item()}

def lossAnharmonicity(alpha):
    def lossFunction(Spectrum,flux_profile):
        anharmonicity = tensor(0.0)
        for spectrum,state in Spectrum:
            anharmonicity += anHarmonicity(spectrum)
        anharmonicity = anharmonicity/len(Spectrum)
        loss = MSE(anharmonicity,tensor(alpha))
        return loss,{'anharmonicity':anharmonicity.detach().item()}
    return lossFunction

def lossE10(E10):
    def lossFunction(Spectrum,flux_profile):
        loss = tensor(0.0)
        spectrum = stack([spectrum[:2] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        loss += MSE(e10,tensor(E10))
        return loss,dict()
    return lossFunction

def lossTransition(E10,E20):
    def lossFunction(Spectrum,flux_profile):
        spectrum = stack([spectrum[:3] for spectrum,state in Spectrum])
        e10 = spectrum[:,1]-spectrum[:,0]
        e20 = spectrum[:,2]-spectrum[:,0]
        loss = MSE(e10,E10) + MSE(e20,E20)
        log = {'mid10':e10[int(len(flux_profile)/2)].detach().item(),'mid20':e20[int(len(flux_profile)/2)].detach().item()}
        return loss,log
    return lossFunction

# Stochastic sample on flux_profile
#def loss_Anharmonicity(Spectrum,flux):
#    spectrum,state = Spectrum
#    return MSE(anHarmonicity(spectrum),tensor(.5))

if __name__=='__main__':
    #torch.set_num_threads(12)
    basis = [1500]
    cuda0 = torch.device('cuda:0')
    torch.set_default_device(cuda0)
    from models import transmon
    import torch.distributed as dist
    from torch.distributed import TCPStore
    #store = TCPStore('localhost',12345)
    #dist.init_process_group(backend=DISTRIBUTION_BACKEND,world_size=1,rank=0,store=store)
    from DiSuQ.Torch.components import DISTRIBUTION_BACKEND
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.nn.parallel import DataParallel as DP
    from circuit import Charge,Kerman
    circuit = transmon(Charge,basis,sparse=False)
    flux_profile = [dict()]
    flux_profile = Tensor([[]])
    control_iDs = dict()
    module = DP(circuit,[cuda0],cuda0)
    loss = lossTransition(tensor(5.),tensor(4.5))
    optim = GradientDescent(circuit,module,flux_profile,loss)
    optim.optimizer = optim.initAlgo(lr=1e-1)
    print(circuit.circuitComponents())  
    dLogs,dParams,dCircuit = optim.optimization(100)
    import ipdb;ipdb.set_trace()
    optim = Minimization(circuit,module,flux_profile,loss)
    dLogs,dParams,dCircuit = optim.optimization()
    print("refer Optimization Verification")
