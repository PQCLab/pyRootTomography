import numpy as np
from scipy.linalg import null_space
from typing import Union
from warnings import warn
from root_tomography.entity import State, Process
from root_tomography.experiment import Experiment, nshots_divide
from root_tomography.tools import extend


def infomatrix(entity: Union[State, Process], ex: Experiment, rank: Union[str, int] = "entity"):
    dm = entity.dm
    dm_norm = np.trace(dm)
    if rank == "entity":
        psi = entity.root
    else:
        psi = State.purify(dm, rank)

    # Find close state with no zeros probabilities
    prob = []
    p_tol = 1e-10
    n_tries = 100
    for j in range(n_tries):
        prob = ex.get_probs_sq(psi)
        if np.all(prob > p_tol):
            break
        if j == n_tries - 1:
            warn("Failed to find non-singular state")
        else:
            psi += (np.random.normal(size=psi.shape) + 1j * np.random.normal(size=psi.shape)) * np.sqrt(p_tol)
            psi = psi / np.sqrt(np.trace(psi.conj().T @ psi) / dm_norm)

    # Calculate Fisher information matrix
    h = 0
    operators = [extend(elem, 3) for elem in ex.proto]
    operators = np.concatenate(tuple(operators), axis=0)
    nshots = ex.vec_nshots
    for elem, n, p in zip(operators, nshots, prob):
        a = np.reshape(elem @ psi, (-1,), order="F")
        a = np.concatenate((np.real(a), np.imag(a)))
        h = h + ex.stat().fisher_information(n, p) * np.outer(a, a)
    h = 4 * h
    return h


def bound(entity: Union[State, Process], ex: Experiment, rank: Union[str, int] = "entity"):
    dm = entity.dm
    if rank == "entity":
        rank = entity.rank
        psi = entity.root
    else:
        psi = State.purify(dm, rank)
    h = infomatrix(entity, ex, rank=rank)

    constraints = []
    sh, uh = np.linalg.eigh(h)

    # Normalization constraints
    if type(entity) is State:
        psi_vec = np.reshape(psi, (-1,), order="F")
        constraints.append(np.concatenate((np.real(psi_vec), np.imag(psi_vec))))
    elif type(entity) is Process:
        pass

    # Phase insensitivity constraints
    tol = max(sh) * 1e-10
    idx = np.where(sh < tol)[0]
    if len(idx) > rank ** 2:
        warn("Information matrix has more than r^2 zero eigenvalues")
    sh[idx] = tol
    idx = idx[:rank ** 2]
    constraints += [uh[:, j] for j in idx]

    # Find variances
    constraints = [extend(constraint, 2).T for constraint in constraints]
    constraints = np.concatenate(tuple(constraints), axis=1)
    q = null_space(constraints.T)
    var_sq = q.T @ uh @ np.diag(1 / np.sqrt(sh))
    var = np.linalg.svd(var_sq, compute_uv=False) ** 2
    fid_d = var / np.real(np.trace(entity.dm))
    return fid_d


def lossfun(entity: Union[State, Process], ex: Experiment, *args, **kwargs):
    if ex.nshots is None:
        ex.set_data(nshots=nshots_divide(1, len(ex.proto), "total"))
    df = sum(bound(entity, ex, *args, **kwargs))
    return df * sum(ex.nshots)
