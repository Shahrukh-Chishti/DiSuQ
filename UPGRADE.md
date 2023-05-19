The combination of techniques employed in DiSuQ provide significant computation advantages.
There are several improvement suggested for further upgradation.

### LOBPCG - Bloc size
* LOBPCG is implemented via torch.eigh
* Bloc size defines the lowest energy bundle, that is solved for Rayliegh minimization
* Quantum mechanics is relevent to bottom 3 levels
* The control over bloc size should mitigate computation resource
* Further, sequential calculation of the ground, Ist and IInd excited levels would discard sorting

$H |g\rangle = \lambda_g |g\rangle$  \  
$H = \sum_{i=0} \lambda_i |i\rangle \langle i|$ , with bloc size = 1, ground state of $H$ is estimated, via Rayleigh. \
$H' \equiv H - \lambda_0 |0\rangle \langle 0|$, ground state of $H'$ is calculated similarly and represent the first excited state.
