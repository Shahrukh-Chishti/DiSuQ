# DiSuQ
Discovering Superconducting Quantum circuits

## Automated Circuit Design
* https://science.sciencemag.org/content/352/6281/aac7341
* https://www.nature.com/articles/d41586-018-07662-w

# Analytic Ground State
Truncated approximation to ground state of Hamiltonian. It is required be functionally differentiable.
In the context of computational graph, it draws parallel from recurrent computation networks.

Approximation builds on Hessian-order optimisation of characterisitic polynomial(squared).
With sufficient iteration, and justified starting point, the algorithm is bound to minimal solution.

## Methods
* Min-max theorem
* https://www.pugetsystems.com/labs/hpc/PyTorch-for-Scientific-Computing---Quantum-Mechanics-Example-Part-2-Program-Before-Code-Optimizations-1222/#define-the-energy-loss-function
* BFGS
- https://towardsdatascience.com/bfgs-in-a-nutshell-an-introduction-to-quasi-newton-methods-21b0e13ee504
- https://stackoverflow.com/questions/42424444/scipy-optimisation-newton-cg-vs-bfgs-vs-l-bfgs
- https://medium.com/swlh/optimization-algorithms-the-newton-method-4bc6728fb3b6
* numpy & scipy
- https://scipy-lectures.org/advanced/mathematical_optimization/
- http://www.cs.columbia.edu/~amoretti/smac_04_tutorial.html
- https://en.wikipedia.org/wiki/Eigenvalue_algorithm#:~:text=The%20eigenvalues%20of%20a%20Hermitian,only%20if%20A%20is%20symmetric.

* LOBPCG[https://pytorch.org/docs/stable/generated/torch.lobpcg.html]
* https://epubs.siam.org/doi/abs/10.1137/17M1129830

 

### Attempts
* Generalized Eigenvalue Problem Derivatives[https://jackd.github.io/posts/generalized-eig-jvp/]
* https://arxiv.org/pdf/2001.04121.pdf
* https://github.com/buwantaiji/DominantSparseEigenAD
### Automatic Differentitation
* https://journals.aps.org/pra/abstract/10.1103/PhysRevA.95.042318
* https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031041 
