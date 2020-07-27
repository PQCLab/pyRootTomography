# Root approach quantum tomography

Python library for the discrete variables quantum state tomography using root approach. The library contains a set of routines for quantum state reconstruction by the complementary measurements results and estimation of statistical adequacy.

## Installation
```
pip install root-tomography
```


## Data format

Measurement protocol is defined by a set of complementary measurement experiments over density matrix ![rho](https://latex.codecogs.com/svg.latex?%5Crho). Every experiment is repeated many times and has several possible measurement outcomes. The probability to get ![k](https://latex.codecogs.com/svg.latex?k)-th outcome is determined by the measurement operator ![M_k](https://latex.codecogs.com/svg.latex?M_k) as ![p_k=trace(rho*M_k)](https://latex.codecogs.com/svg.latex?p_k%3D%5Ctext%7BTr%7D%28%5Crho%20M_k%29). The set of measurement operators and the number of experiments repetitions define the **_measurement protocol_**. The number of observations for each outcome define the **_measurement results_**. The following code describe the required data format.
```
proto[j][k,:,:]  # Measurement operator matrix for k-th outcome in j-th experiment
nshots[j]  # Number of j-th experiment repetitions
clicks[j][k]  # Number of k-th outcome observations in j-th experiment
```

The following code generates an example of ideal ![N](https://latex.codecogs.com/svg.latex?N)-qubit factorized MUB protocol with every measurement experiments repeated ![n](https://latex.codecogs.com/svg.latex?n) times.
```
import root_tomography.tools as tools
N = 1  # Number of qubits
n = 1e3  # Number of repetitions in every experiment
proto = tools.protocol("MUB", [2]*N)
nshots = tools.nshots_devide(n, len(proto), "equal")
```

One can simulate the measurements of some density matrix.
```
dm_true = tools.gen_state("haar_dm", 2, rank=1)  # Generate Haar-random single-qubit pure state density matrix
clicks = tools.simulate(dm_true, proto, nshots)  # Simulate measurements
```

## Quantum state reconstruction
Reconstruct density matrix and estimate fidelity comparing to true state.
```
from root_tomography.reconstruction import dm_reconstruct
dm, _ = dm_reconstruct(clicks, proto, nshots, display=True)
print("Fidelity: {}".format(tools.fidelity(dm, dm_true)))
```
Output:
```
=== Automatic rank estimation ===
Try rank 1
Iteration 19      Difference 9.3151e-09
Rank 1 is statistically significant at significance level 0.05. Procedure terminated.
Fidelity: 0.9999793077308207
```
If one don't specify `nshots` then number of ![j](https://latex.codecogs.com/svg.latex?j)-th experiment runs is taken as the sum over all counts.
```
dm_reconstruct(clicks, proto)  # same as nshots[j] = sum(clicks[j])
```
Instead of automatic rank estimation one can pick a specific rank of the quantum state model. For example, in some experiments it could be likely to have a pure (rank 1) state.
```
dm, _ = dm_reconstruct(clicks, proto, nshots, rank=1, display=True)
print("Fidelity: {}".format(tools.fidelity(dm, dm_true)))
```
Output:
```
Iteration 16      Difference 7.9310e-09
Fidelity: 0.9997600124722302
```
In general, you can specify any rank from 1 to the Hilbert space dimension.

## Algorithms

Consider a quantum state in the Hilbert space ![H](https://latex.codecogs.com/svg.latex?%5Cmathcal%7BH%7D) described by the density matrix ![rho](https://latex.codecogs.com/svg.latex?%5Crho).
According to the root approach the rank-![r](https://latex.codecogs.com/svg.latex?r) quantum state parametrization is defined by the complex matrix ![c](https://latex.codecogs.com/svg.latex?c) of size ![sxr](https://latex.codecogs.com/svg.latex?s%20%5Ctimes%20r) (![s=dimH](https://latex.codecogs.com/svg.latex?s%3D%5Ctext%7Bdim%7D%5Cmathcal%7BH%7D)) such that ![rho=cc+](https://latex.codecogs.com/svg.latex?%5Crho%3Dcc%5E%5Cdagger). To get the quantum state maximum likelihood estimation one must solve the following quasi-linear equation (_likelihood equation_) [[1]](#ref1):
<p align="center"><img src="https://latex.codecogs.com/svg.latex?Ic%3DJ%28c%29c"/></p>
where
<p align="center"><img src="https://latex.codecogs.com/svg.latex?I%3D%5Csum_%7Bj%7D%7Bn_j%20M_j%7D%2C%5C%3B%5C%3B%5C%3B%5C%3BJ%28c%29%3D%5Csum_%7Bj%7D%7B%5Cfrac%7Bk_j%7D%7B%5Ctext%7BTr%7D%28cc%5E%5Cdagger%20M_j%29%7D%20M_j%7D"/></p>

The sums here are taken over all measurement experiments and possible outcomes in them. ![k_j](https://latex.codecogs.com/svg.latex?k_j) is the number of observed outcomes corresponding to the measurement operator ![M_j](https://latex.codecogs.com/svg.latex?M_j) and the number of measurements repetitions ![n_j](https://latex.codecogs.com/svg.latex?n_j).

The search of the likelihood equation solution is performed by the fixed-point iteration method:
<p align="center"><img src="https://latex.codecogs.com/svg.latex?c_%7Bi&plus;1%7D%3D%281-%5Calpha%29I%5E%7B-1%7DJ%28c_i%29c_i%20&plus;%20%5Calpha%20c_i"/></p>

Here ![alpha](https://latex.codecogs.com/svg.latex?%5Calpha) is the regularization parameter. We use Moore-Penrose pseudo-inversion to get ![c_0](https://latex.codecogs.com/svg.latex?c_0).

## References

<a name="ref1">[1]</a> Bogdanov Yu. I. Unified statistical method for reconstructing quantum states by purification // _JETP_ **108(6)** 928-935 (2009); doi: 10.1134/S106377610906003X
