# DiSuQ
Discovering Superconducting Quantum circuits.

This library is aimed to provide optimization and solution to general lumped-model superconducting circuits. 


## Problem - Analytic Ground State
Truncated approximation to ground state of Hamiltonian. It is required be functionally differentiable.
In the context of computational graph, it draws parallel from recurrent computation networks.

Approximation builds on Hessian-order optimisation of characterisitic polynomial(squared).
With sufficient iteration, and justified starting point, the algorithm is bound to minimal solution.

# Modular Structure
Backend design of DiSuQ is maintained felxible to allow computation over full range of resources.

### Layered Hierarchy
Classes and functional modules are partitioned over mathematical operations. The APIs thus allow to extend this framework to diverse range of quantum system.

# Features
## Autograd
Several advantages are inherited from the Torch backend:
* Analytic differentiation of Eigen-decomposition
* Exact gradient calculations : avoiding difference approximates 
* CUDA supported matrix operations : GPU acceleration
* Distributed computation: High Performance architecture scalability
* Backpropagation : 
    - Faster evaluation of parameter gradients
## Backend


## Extension
Faster gradient calculation enable to extend modeling of SuperCond circuits to a larger space of parameterized effective description:
- Highly parameteric models:
    * Element design: Generalized lumped modeling
    * Dissipative elements: Modeling noise & decoherence
        - Mixed state preparation : Noise controling
    * Parametric basis evaluation

- Basis description: Subclass modularity allows to design custom heuristic circuit quantization. Example:`Kerman`. 
- Extension of `Circuit` class to other quantum systems
    * Processors: NV-centers, trapped Ions etc.
    * Modules inheritance: `Components`,`Optimization`,`Distribution` maintain their utility for simulation/optimization.

- Smooth variation in the distribution of Flux,Charge

# Notes
* Degeneracy Vicinity:
    - In the vicinity of degeneracy $\lambda_i - \lambda_j \to 0$, the eigvalsh should be prefered to eigh[torch](https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html).
