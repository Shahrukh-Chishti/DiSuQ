DiSuQ APIs(application programming interface) is structured in its objective:
- Simulating Superconducting circuit quantum physics
- Optimization cirucit on specified Objectivity

# Circuit

## Network

## Initialization

# Simulation

# Optimization

* Control Operating Point:
    - loop flux:
        - control_iD: `str`: inductive element
    - node bias:
        - control_iD: `int`: circuit node

## Distributed
* Spawning is preferable on python interpreter(script execution)
* Multi-processing program in Jupyter iPython is generally unfavourable
* Spawning is non-functionary on Jupyter pickle serializing

Reference code : [parallel.py](https://github.com/Shahrukh-Chishti/DiSuQ/blob/ad093038729fa427beb87f844c3ec48ff9d49b7b/Torch/parallel.py)
1. Circuit, Control, Optimizer
2. Distributor
3. Multiprocess Spawning
