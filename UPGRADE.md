The combination of techniques employed in DiSuQ provide significant computation advantages.
Three fundamental computation procedure for this programme are:
* Circuit quantization
* Eigen-decomposition
* Optimization

There are several improvement suggested for further upgradation.

# Hamiltonian Quantization
* exponentiation operators sparse construction
* Kerman transformation to sparse, integer fields
* `performance` branch : persist Displacement(Z) operators

# Eigendecomposition Algorithm
## Partial Diagonalization
* Extending partial circuit diagonalization over multi-GPU 
## Incremental Diagonalization
Middle region circuit diagonalization
### Circuit calculus - Projective
* circuit element bulking
* additional degree of freedom
### Basis Rotation
* traversing inside in smaller truncated subspace
* Trotterized rotation of bases along the path

Comparison between partial & incremental:
* computation expense
* accuracy bounds

## LOBPCG - Bloc size
* LOBPCG is implemented via torch.eigh
* Bloc size defines the lowest energy bundle, that is solved for Rayliegh minimization
* Quantum mechanics is relevent to bottom 3 levels
* The control over bloc size should mitigate computation resource
* Further, sequential calculation of the ground, Ist and IInd excited levels would discard sorting

$H |g\rangle = \lambda_g |g\rangle$  \  
$H = \sum_{i=0} \lambda_i |i\rangle \langle i|$ , with bloc size = 1, ground state of $H$ is estimated, via Rayleigh. \
$H' \equiv H - \lambda_0 |0\rangle \langle 0|$, ground state of $H'$ is calculated similarly and represent the first excited state.

# Optimization

## Multi-Init discovery
Gradient descent methods are strongly converging under convex conditions. It might happen that during multi-init trajectory exploration ie. diverse starting point, many trajectory fall under same basin of attraction. The calculations would be redundant.

# Backend Design

## Static & Control Hamiltonian
Rebuilding Hamiltonian for each new control setting, from the ground-up is redundant. Hamiltonian maybe additively decomposed into static and dynamic subparts. Persisting the static component saves computation time at the expense of memory.
Similar to the approach of `performance` branch, backends with flexibility in persistence could be designed.
Instead of persisting operators, hamiltonian could be decomposed on the basis of variation, for a given utilization.
$$
H = H_{static} + H_{control}(\theta)
$$
Usually, the $H_{control}$ is factorizable wrt. controls $\theta$.
Re-calculation would still preserve the computation graph for circuit parameters.
