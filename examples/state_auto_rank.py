import matplotlib.pyplot as plt
from root_tomography.experiment import Experiment, proto_measurement
from root_tomography.entity import State
from root_tomography.estimator import reconstruct_state
from root_tomography.bound import bound
from root_tomography import gchi2

# Experiment conditions
dim = 3
r_true = 1
nshots = int(1e3)
proto = proto_measurement("mub", dim=dim)
# proto = proto_measurement("tetra", modifier="operator")  # Uncomment to test Poisson stats

# Generate state and data
state_true = State.random(dim, r_true)
clicks = Experiment(dim, State).set_data(proto=proto, nshots=nshots).simulate(state_true.dm)

# Reconstruct state and compare to the true one
state_rec = reconstruct_state(dim, clicks, proto, nshots, display=10)[0]
fid = State.fidelity(state_true, state_rec)
print(f"Fidelity: {fid:.6f}")

# Calculate fiducial fidelity bound
d = bound(state_rec, proto, nshots)
fid95 = 1 - gchi2.ppf(d, 0.95)
print(f"Fiducial 95% fidelity bound: {fid95:.6f}")

# Plot theoretical infidelity distribution
d = bound(state_true, proto, nshots)
p, df = gchi2.pdf(d)
plt.figure("Infidelity")
plt.plot(df, p, label="Theory")
plt.plot([1-fid, 1-fid], [0, max(p)*1.05], label="Reconstruction")
plt.xlabel("infidelity")
plt.legend()
plt.show()
