import matplotlib.pyplot as plt
from root_tomography.experiment import Experiment, proto_measurement
from root_tomography.entity import State
from root_tomography.estimator import reconstruct_state
from root_tomography.bound import bound
from root_tomography import gchi2

# Experiment conditions
n_exp = 500
dim = 2
r_true = 1
r_rec = 1
nshots = int(1e3)
proto = proto_measurement("mub", dim=dim)
# proto = proto_measurement("tetra", modifier="operator")  # Uncomment to test Poisson stats

# Generate state
state_true = State.random(dim, r_true)

# Conduct experiments
infid = []
pval = []
for j in range(n_exp):
    print(f"Experiment {j+1:d}/{n_exp:d}")
    clicks = Experiment(dim, State).set_data(proto=proto, nshots=nshots).simulate(state_true.dm)

    state_rec, rinfo = reconstruct_state(dim, clicks, proto, nshots, rank=r_rec, get_stats=True)
    infid.append(1 - State.fidelity(state_true, state_rec))
    pval.append(rinfo["pval"])

# Plot histograms
plt.figure("P-value")
plt.hist(pval, edgecolor="black", density=True, label="Numerical Experiments")
plt.plot([0, 1], [1, 1], label="Theory")
plt.xlabel("p-value")
plt.legend()
plt.show()

plt.figure("Infidelity")
plt.hist(infid, edgecolor="black", density=True, label="Numerical Experiments")
d = bound(state_true, proto, nshots)
p, df = gchi2.pdf(d)
plt.plot(df, p, label="Theory")
plt.xlabel("infidelity")
plt.legend()
plt.show()


