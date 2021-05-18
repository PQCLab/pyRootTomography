import numpy as np
from scipy.special import gammaincc
from itertools import product
from root_tomography.entity import State


def meas_matrix(proto):
    if type(proto) is list:
        proto = [extend(p, 3) for p in proto]
        proto = np.concatenate(tuple(proto), axis=0)
    return np.reshape(proto, (proto.shape[0], -1))


def kron3d(a1: np.ndarray, a2: np.ndarray):
    a1 = extend(a1, 3)
    a2 = extend(a2, 3)
    s1 = np.array(a1.shape)
    s2 = np.array(a2.shape)
    a = np.empty(tuple(s1 * s2), dtype=complex)
    for j1, j2 in product(range(s1[0]), range(s2[0])):
        a[j1 * s2[0] + j2, :, :] = np.kron(a1[j1, :, :], a2[j2, :, :])
    return a


def extend(a: np.ndarray, dims: int):
    s = list(a.shape)
    if len(s) < dims:
        s = [1] * (dims - len(s)) + s
        a = np.reshape(a, tuple(s))
    return a


def base2povm(base):
    d = base.shape[0]
    m = base.shape[1]
    povm = np.empty((m, d, d), dtype=complex)
    for j in range(m):
        povm[j, :, :] = np.outer(base[:, j], base[:, j].conj())
    return povm


def pinv(x, probs, rank=None):
    bmat = meas_matrix(x) if type(x) is list else x
    dim = int(np.sqrt(bmat.shape[1]))
    dm = np.reshape(np.linalg.lstsq(bmat, probs, rcond=None)[0], (dim, dim), order="F")
    sq = State.purify(dm, rank)
    return State.from_root(sq)


def uprint(text, nb=0, end=""):
    textsp = text + " "*max(0, nb-len(text))
    print("\r" + textsp, end=end, flush=True)
    return len(textsp)


def pval(chi2: float, df: int):
    return gammaincc(df/2, chi2/2) if df > 0 else np.nan
