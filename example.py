import root_tomography.tools as tools
from root_tomography.reconstruction import dm_reconstruct

N = 1  # Number of qubits
n = 1e3  # Number of repetitions in every experiment
proto = tools.protocol("MUB", [2]*N)
nshots = tools.nshots_devide(n, len(proto), "equal")

dm_true = tools.gen_state("haar_dm", 2**N, rank=1)  # Generate Haar-random single-qubit pure state density matrix
clicks = tools.simulate(dm_true, proto, nshots)  # Simulate measurements

dm, _ = dm_reconstruct(clicks, proto, nshots, display=True)
print("Fidelity: {}".format(tools.fidelity(dm, dm_true)))