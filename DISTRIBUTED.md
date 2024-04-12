To speed up calculation on high dimesional control data,
DiSuQ - `distributed` module enables data parallelization on multi-GPU architecture.

* Its design is similar to `DDP` module from torch.
* API : hierarchical-topmost, plugable layer on `Optimization` objects

Distributed Parallelization procedure:
* Process Distribution:
1. Distribute identical copy of model on each n-GPU
2. Control data is partitioned(not necessarily a single point)
3. Control chunks are exclusively distributed over GPUs 
4. Preferably, control data set $n_{data} = |\{\Phi_i\}|$ is n-factorizable

* Parallel calculations:
1. Each process($i$) calculates set of eigendecomposition
    - eigenvalues $\Lambda$, eigenvectors $\Omega$
    - corresponds to set of control points($i$) 
2. Spectrum is `all_gather` from different GPUs
    - synchronize call
    - broadcast is restricted to data-only
    - the JIT computational graph and gradient dependency is lost
3. Replace the host tensor in the `all_gather` tensor list
    - it contains computational graph for host calculations
    - specific to control data sent to this process
4. Calculate general loss function($\mathcal{L}(\Omega_1, \cdots \Omega_n)$)
    - identical value on each process
5. Compute parameter gradients on each process, independently:
    - backprop follows from leaf-node loss, via local process Spectrum
    - independent gradients $\frac{\partial \mathcal{L}}{\partial \theta}$ accumulate on circuit parameters for each process
6. Distribute and summate gradients among all processes
    - synchronize call
7. Optimization step:
    - Update parameters
    - repeat independently

$$
\frac{\partial \mathcal{L}}{\partial \theta_i} = \sum_j \frac{\partial \mathcal{L}}{\partial \Lamdba_j} \frac{\partial \Lambda_j}{\partial \theta_i}
$$


