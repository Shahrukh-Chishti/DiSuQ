# DiSuQ
Discovering Superconducting Quantum circuits.

This library is aimed to provide optimization and solution to general lumped-model superconducting circuits. 


## Problem - Analytic Ground State
Truncated approximation to ground state of Hamiltonian. It is required be functionally differentiable.
In the context of computational graph, it draws parallel from recurrent computation networks.

Approximation builds on Hessian-order optimisation of characterisitic polynomial(squared).
With sufficient iteration, and justified starting point, the algorithm is bound to minimal solution.

## Modular Structure
Backend design of DiSuQ is maintained felxible to allow computation over wide range of resources.

### Layered Hierarchy
Classes and functional modules are partitioned over mathematical operations. The APIs thus allow to extend this framework to diverse range of quantum system.

## Features
### Basis
Numerical computation over operator algebra requires basis selection. Equivalent to choice of coordinate system, it should appropriate given quantum system.
In our opinion, its a long-standing problem to identify optimal basis of representation, depending on the location in circuit parameter space.

DiSuQ `Circuit` provides various choice in basis selection:
1. Kerman
2. Charge
3. Oscillator
4. Flux
5. Mixed

### Autograd
Several advantages are inherited from the Torch backend:
* Analytic differentiation of Eigen-decomposition
* Exact gradient calculations : avoiding difference approximates 
* CUDA supported matrix operations : GPU acceleration
* Distributed computation: High Performance architecture scalability
* Backpropagation : 
    - Faster evaluation of parameter gradients

### Backend
DiSuQ allows two levels of backend:
1. Data Structure : 
    - `numpy` v/s `torch` encoding circuit operator representation, 
    - `numpy` implementation is to be deprecated
2. Data Encoding:
    - `sparse`
        * `torch.LOBPCG` is unavailable
    - `dense`
        * memory-challenging
        * full availability of `linalg` subroutines
3. Initialization : 
    - currently, distributed over branches : `performance` and `main`
    - initialization v/s run-time calculation of Hermitian matrix operator matrices 
    - pre-calculated matrices offer faster revaluation
    - run-time calculation saves multiples of memory

Intialization choice is subjective to utilization. Depending on the application, and problem size initialization backend should be decided.

For example, a small sized circuit requiring high amount of reinitialization, in optimization or control-landscape calculation could be persisted in the memory.
It would save the overhead of creating operator `tensor`. Another utilization could be repeated operator expectation over eigenstates. 

However, if the circuit is memory-demanding, then pre-occupying memory for various operator `tensor` is disadvantageous.

*Kronecker computation is time-consuming calculation*

### Extension
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

## Notes
* Degeneracy Vicinity:
    - In the vicinity of degeneracy $\lambda_i - \lambda_j \to 0$, the eigvalsh should be prefered to eigh[torch](https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html).
