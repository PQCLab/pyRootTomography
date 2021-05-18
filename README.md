# Root approach quantum tomography

Python library for the discrete variables quantum state tomography using root approach and maximum likelihood estimation. The library contains a set of tools for quantum state reconstruction by the complementary measurements results, estimation of statistical adequacy and theoretical analysis of reconstruction fidelity.

- [Installation](#installation)
- [Definitions](#definitions)
- [Algorithm](#algorithm)
- [Data format](#format)
- [Quantum state tomography](#qst)
- [References](#references)
- [License](#license)

## <a name="installation">Installation</a>
```
pip install root-tomography
```

## <a name="definitions">Definitions</a>
Consider a quantum state in the Hilbert space of dimension <img alt="d" src="https://render.githubusercontent.com/render/math?math=d" />. The root approach to quantum state tomography implies reconstructing a purified quantum state <img alt="psi" src="https://render.githubusercontent.com/render/math?math=\psi" /> of size <img alt="d-by-r" src="https://render.githubusercontent.com/render/math?math=d\times r" /> instead of corresponding rank-<img alt="r" src="https://render.githubusercontent.com/render/math?math=r" /> **density matrix** <img alt="rho=psi*psi" src="https://render.githubusercontent.com/render/math?math=\rho=\psi\psi^\dagger" />. Thus, matrix <img alt="psi" src="https://render.githubusercontent.com/render/math?math=\psi" /> defines a **square root** of the density matrix.

We measure the reconstruction accuracy by Uhlmann's **fidelity** between the true state <img alt="rho" src="https://render.githubusercontent.com/render/math?math=\rho" /> and the reconstructed state <img alt="sigma" src="https://render.githubusercontent.com/render/math?math=\sigma" />:
<p align="center"><img alt="fidelity" src="https://render.githubusercontent.com/render/math?math=\displaystyle F(\rho,\sigma)=\left[\textrm{Tr}\sqrt{\sqrt\rho\sigma\sqrt\rho}\right]^2." /></p>

According to the quantum state estimation theory the **infidelity distribution** is bounded by the general chi-squared distribution with <img alt="nu" src="https://render.githubusercontent.com/render/math?math=\nu" /> degrees of freedom [[1]](#ref1):
<p align="center"><img alt="infidelity as random variable" src="https://render.githubusercontent.com/render/math?math=\displaystyle 1-F\sim\sum_{j=1}^{\nu}{d_j\xi_j^2}," /></p>

where <img alt="d_j" src="https://render.githubusercontent.com/render/math?math=d_j" /> are positive parameters and <img alt="xi_j" src="https://render.githubusercontent.com/render/math?math=\xi_j" /> are independent random variables with standard normal distribution. The expected value and variance of infidelity are thus
<p align="center"><img alt="infidelity mean and variance" src="https://render.githubusercontent.com/render/math?math=\displaystyle \langle1-F\rangle=\sum_{j=1}^{\nu}{d_j}, \qquad \sigma_{1-F}^2=2\sum_{j=1}^{\nu}{d_j^2}." /></p>

As the infidelity lower bound is inverse proportional to the total sample size <img alt="N" src="https://render.githubusercontent.com/render/math?math=N" /> over all measurements, we also use the so-called **loss function** <img alt="L=N*mean(1-F)" src="https://render.githubusercontent.com/render/math?math=L=N\langle1-F\rangle" /> independent of the sample size.

## <a name="algorithm">Algorithm</a>
We use the _maximum likelihood_ parameters estimation (MLE). In the case of quantum state tomography MLE results in the _likelihood equation_ [[1]](#ref1). Taking the density matrix normalization condition, one obtains
<p align="center"><img alt="likelihood equation" src="https://render.githubusercontent.com/render/math?math=\displaystyle \mu\psi=J(\psi\psi^\dagger)\psi," /></p>

where <img alt="mu" src="https://render.githubusercontent.com/render/math?math=\mu" /> is a constant (usually equal to the total observed counts <img alt="sum(k_j)" src="https://render.githubusercontent.com/render/math?math=\sum_j{k_j}" />) and
<p align="center"><img alt="J definitions" src="https://render.githubusercontent.com/render/math?math=\displaystyle J(\rho)=b_0(\rho) I %2B \sum_j{b_j(\rho)P_j}." /></p>

The sums are taken over all measurements operators in all measurements schemes. The actual values of <img alt="b_0" src="https://render.githubusercontent.com/render/math?math=b_0" /> and <img alt="b_j" src="https://render.githubusercontent.com/render/math?math=b_j" /> depend on the measurements statistics (see [Measurements statistics](#statistics) section).

An equation solution is obtained by the fixed-point iteration method:
<p align="center"><img alt="fixed-point iterations" src="https://render.githubusercontent.com/render/math?math=\displaystyle \psi_{i%2B1}=(1-a)\mu^{-1}J(\psi\psi^\dagger)\psi_i %2B a\psi_i." /></p>

Here <img alt="a" src="https://render.githubusercontent.com/render/math?math=a" /> is the regularization parameter. We use Moore-Penrose pseudo-inversion to get <img alt="psi_0" src="https://render.githubusercontent.com/render/math?math=\psi_0" />.

The root approach quantum tomography implies setting the model rank <img alt="r" src="https://render.githubusercontent.com/render/math?math=r" />. If the rank is unknown we estimate it using the _adequacy criterion_ [[1]](#ref1). To do this we vary <img alt="r" src="https://render.githubusercontent.com/render/math?math=r" /> from 1 to its maximal value until the reconstruction result becomes statistically significant at some pre-chosen significance level. The procedure is also terminated if the p-value of the rank-<img alt="(r+1)" src="https://render.githubusercontent.com/render/math?math=(r%2B1)" /> model is lower than p-value of the rank-<img alt="r" src="https://render.githubusercontent.com/render/math?math=r" /> model.


## <a name="format">Data format</a>
For the **quantum state tomography** one must specify a set of complementary measurement experiments over a quantum state density matrix <img alt="rho" src="https://render.githubusercontent.com/render/math?math=\rho" />. Every experiment may be repeated many times with some sets of possible measurement outcomes. The probability to get _k_-th outcome is determined by the measurement operator <img alt="P_k" src="https://render.githubusercontent.com/render/math?math=P_k" /> as <img alt="p_k=trace(rho*P_k)" src="https://render.githubusercontent.com/render/math?math=p_k=\textrm{Tr}(\rho P_k)" />. The set of measurement operators and the number of experiments repetitions define the **_measurements protocol_**. The number of observations for each outcome define the **_measurements results_**. The following code describe the required data format.
```python
proto  # List of measurement operators matrices
proto[j]  # 3d numpy array of measurement operator matrices in j-th measurement scheme
proto[j][k, :, :]  # Measurement operator matrix for k-th outcome in j-th measurement scheme
```
```python
nshots  # List of measurement schemes repetitions
nshots[j]  # Number of j-th scheme repetitions
```
```python
clicks  # List of outcomes
clicks[j]  # 1d numpy array of measurement outcomes in j-th scheme
clicks[j][k]  # Number of k-th outcome observations in j-th scheme
```
One can specify `nshots` as a single integer describing the total sample size. Then it is automatically divided equally over all measurement schemes.

One can also pass a 3d numpy array for measurement operators and a 1d array for measurements results implying only a single possible outcome in each measurement:
```python
proto[j, :, :]  # Measurement operator matrix for j-th measurement scheme
nshots[j]  # Number of j-th scheme repetitions
clicks[j]  # Number of observations in j-th scheme
```

## <a name="qst">Quantum state tomography</a>

Examples directory of the project contains a set of examples that show basic features of the library. Below we briefly review the quantum state tomography example.

Consider an example of a pure ququart state reconstruction using mutually-unbiased bases measurement protocol.
```python
from root_tomography.experiment import proto_measurement
from root_tomography.entity import State

dim = 4  # System dimension
r_true = 1  # True state rank
state_true = State.random(dim, r_true)  # True state
nshots = 10 ** 3  # Total sample size
proto = proto_measurement("mub", dim=dim)  # Generate measurements operators
```

The `Experiment` class of `experiment` module allows one to store the tomography data.
```python
from root_tomography.experiment import Experiment

ex = Experiment(dim, State)  # Generate experiment instance
ex.set_data(proto=proto, nshots=nshots)  # Set measurements protocol
ex.set_data(clicks=clicks)  # Set measurements data
```

One can also perform the measurement simulation.
```python
ex.simulate(state_true.dm)  # Performs the simulation and stores it
```


The reconstruction is performed using the `reconstruct_state` function of `estimator` module. By default the state rank is estimated automatically using the adequacy criteria.
```python
from root_tomography.estimator import reconstruct_state

state_rec = reconstruct_state(ex, display=10)
fid = State.fidelity(state_true, state_rec)
print(f"Fidelity: {fid:.6f}")
```

Output:
```
=== Automatic rank estimation ===
=> Try rank 1
Optimization: fixed point iteration method
Iteration 37 		 Delta 9.42e-09
=> Rank 1 is statistically significant at significance level 0.05000. Procedure terminated.
Fidelity: 0.995364
```

Using the fiducial approach and the theoretical infidelity distribution one can use the reconstruction result to estimate the guaranteed reconstruction fidelity. In the following code we estimate the 95%-probability fidelity bound ![F_95](https://latex.codecogs.com/svg.latex?F_%7B95%7D). That means that we get the fidelity ![F_95](https://latex.codecogs.com/svg.latex?F_%7B95%7D) or higher with probability 95%.
```python
from root_tomography.bound import bound
from root_tomography import gchi2

d = bound(state_rec, ex)  # Calculate variances
fid95 = 1 - gchi2.ppf(d, 0.95)  # Get fidelity bound
print(f"Fiducial 95% fidelity bound: {fid95:.6f}")
```

Output:
```
Fiducial 95% fidelity bound: 0.992822
```

The following code plots the infidelity distribution based on the true state and shows the fidelity of reconstructed state.
```python
import matplotlib.pyplot as plt

d = bound(state_true, ex)
p, df = gchi2.pdf(d)
plt.figure("Infidelity")
plt.plot(df, p, label="Theory")
plt.plot([1-fid, 1-fid], [0, max(p)*1.05], label="Reconstruction")
plt.xlabel("infidelity")
plt.legend()
plt.show()
```

![Theoretical distribution and reconstruction result](examples/infidelity.png?x=5)

The theoretical infidelity mean and variance could be obtained from the variance vector.
```python
sum(d)  # Mean infidelity
2 * sum(d ** 2)  # Infidelity variance
1-sum(d)  # Mean fidelity
```

## <a name="references">References</a>

<a name="ref1">[1]</a> Bogdanov Yu. I. Unified statistical method for reconstructing quantum states by purification // _JETP_ **108(6)** 928-935 (2009); doi: 10.1134/S106377610906003X

## <a name="license">License</a>

All code found in this repository is licensed under GPL v3
