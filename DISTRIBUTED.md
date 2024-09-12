To speed up calculation on high dimesional control data,
DiSuQ - `distributed` module enables data parallelization on multi-GPU architecture.

* Design is similar to `DDP` module from torch.
* API : hierarchical-topmost, plugable layer on `Optimization` objects

# Distributed Parallelization procedure
## * Process Distribution:
1. Distribute identical copy of model on each n-GPU
2. Control data is partitioned(not necessarily a single point)
3. Control chunks are exclusively distributed over GPUs 
4. Preferably, control data set $n_{data} = | \{ \Phi_i \} | $ is n-factorizable

## * Parallel calculations:
1. Each process($i$) calculates set of eigendecomposition
    - eigenvalues $\Lambda$, eigenvectors $\Omega$
    - corresponds to set of control points($i$) 
2. Spectrum is `all_gather` from different GPUs
    - *synchronize call*
    - broadcast is restricted to data-only
    - the JIT computational graph and gradient dependency is lost
3. Replace the host tensor in the `all_gather` tensor list
    - it contains computational graph for host calculations
    - specific to control data sent to this process
4. Calculate general loss function($ \mathcal{L} ( \Omega_1, \cdots \Omega_n ) $)
    - identical value on each process
5. Compute parameter gradients on each process, independently:
    - backprop follows from leaf-node loss, via local process Spectrum
    - independent gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ accumulate on circuit parameters for each process
6. Distribute and summate gradients among all processes
    - *synchronize call*
7. Optimization step:
    - Update parameters
    - repeat independently

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} = \sum_j \frac{\partial \mathcal{L}}{\partial \Lambda_j} \frac{\partial \Lambda_j}{\partial \theta_i}
$$

# Parallelism

## Simulation
Simulating loss-scape over control data manifold could be simply extended over `distriubted` parallelism.
There is no gradient evaluation and broadcasting.
Its not yet implemented.

## Blocking Parallelism

Since, step 2 & 6 are blocking sync calls, aprior calculations must be balanced among processes.
That is, control data must balanced to approximate equal calculation time on each process.
This is important for efficient step 2.
Between step 2 & 6, loss calculation is minimal, so its mostly in-sync.

## Approach & Extension 

* The premise for Data-based parallelism is that the model(circuit) evaluation is mostly sequential
    - especially with circuit calculus and basis alignment 
* However, for really large circuits, high capacity could be checked
    - with partial diagonalization, sub-circuits could be process-distributed
    - sub-circuit [pipelining](https://pytorch.org/docs/stable/pipeline.html) could be implemented
    - utilizing island-interaction method, the difference in Hilbert size and consequently the difference in computation time would block parallelization
* Despite, under different convergence criterion eigenvalue decomposition alogrithms are dynamically termination and would not be completely in-sync
* Unlike, StochasticGradient method, control data for SuperCircuit optimization are not randomly generated
* Minimal Duplicacy : loss function calculation is identical on each process


# Distributor
`Distributor` object handles meta-data and distribution protocols to distribute optimization.
* circuit builder wrapper, ex. `DiSuQ.Torch.models`
* control data and distribution profile 

Additionally, a wrapper function `executionSetup`, is required to be custom defined
- identically duplicated over mulitple processes:
* specify loss function
* `Optimizer` and optimization

# Notes
* usage guideline :
    - script interpreter : computation
    - iPython notebooks : evaluation and analysis
* does `distributed.all_gather` maintain `autograd` on host tensor
    - No. Only `tensor.data` is communicated
* `DDP` module from `torch` provides multi-process distributed SGD
    - requires `Circuit` to inherit `nn.Module`
    - to be deprecated
* `torch.DP` module is ineffective for `performance` backend
    - run-time replication inconsistent over multiple devices 
    - pre-initialized tensors are erronuously shared
    - run-time backend is consistent with `DP`
