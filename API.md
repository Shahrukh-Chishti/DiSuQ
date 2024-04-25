DiSuQ APIs(application programming interface) is structured in its objective:
- Simulating Superconducting circuit quantum physics
- Optimization cirucit on specified Objectivity

# Circuit

## Network

## Initialization
* Circuit parameters
* Control Operating Point:
    - loop flux:
        - control_iD: `str`: inductive element
    - node bias:
        - control_iD: `int`: circuit node

# Simulation

# Optimization
`Optimization` class covers optimization of circuits. There two subclasses inheriting common utilization:
1. `Minimization` :
    - `scipy` based optimization methods
    - eigenvalue, derivative evaluation via `torch.autograd` 
2. `GradientDescent`:
    - `torch` based optimization methods

* Multi-initialization :  

## Distributed
* Spawning is preferable on python interpreter(script execution)
* Multi-processing program in Jupyter iPython is generally unfavourable
* Spawning is non-functionary on Jupyter pickle serializing

Reference code : [parallel.py](https://github.com/Shahrukh-Chishti/DiSuQ/blob/ad093038729fa427beb87f844c3ec48ff9d49b7b/Torch/parallel.py)
1. Circuit, Control, Optimizer
2. Distributor
3. Multiprocess Spawning


# Extension Guidelines

Since `torch` is not optimal in computational graph compilation(JIT), it is important to construct calculations strategically.

* minimize copying
* create `tensor` on device
* re-use `tensor`
* sparsity
* gradient calculation switch
* `torch.compile`

- circuit parameter calculations are instantaneous
- does update change intermediate `tensors`, ex.Cinv,Linv : __No__
