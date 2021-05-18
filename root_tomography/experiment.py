import numpy as np
from cmath import exp, pi
from itertools import product
from typing import Union
import root_tomography.meas_statistics as stat
from root_tomography.entity import State, Process, part_trace, rand_unitary
from root_tomography.tools import kron3d, extend, base2povm, meas_matrix
rng = np.random.default_rng()


class Experiment:
    dim = None
    entity = None
    stat_pkg = "auto"
    proto = None
    nshots = None
    clicks = None
    _vec_proto = None
    _vec_nshots = None
    _vec_clicks = None

    def __init__(self, dim: int, entity, stats: Union[str, stat.Statistics] = "auto"):
        self.dim = dim
        self.entity = entity
        if self.entity.__name__ not in ["State", "Process"]:
            raise ValueError(
                "Unknown entity: '{}'\n Only State, Process are available".format(entity.__name__))
        if type(stats) is str:
            stats = stats.lower()
            st_allowed = list(stat.BUILD_IN.keys())
            st_allowed.append("auto")
            if stats not in st_allowed:
                raise ValueError("Unknown statistics type: {}\n Available types: {}".format(stats, ", ".join(st_allowed)))
        self.stat_pkg = stats

    def set_data(self, proto=None, nshots=None, clicks=None):
        if proto is not None:
            if self.entity is Process:
                pass

            if type(proto) is not list:
                proto = [proto[j, :, :] for j in range(proto.shape[0])]
            proto = [extend(elem, 3) for elem in proto]

            self.proto = proto
            self._vec_proto = None

        if clicks is not None:
            if type(clicks) is not list:
                clicks = [clicks[:, j] for j in range(clicks.shape[1])]
            self.clicks = clicks
            self._vec_clicks = None

        if nshots is not None:
            if type(nshots) is np.ndarray:
                nshots = list(nshots)
            self.nshots = nshots
            self._vec_nshots = None

        if self.proto is not None and self.nshots is not None:
            if type(self.nshots) is not list:
                self.nshots = nshots_divide(self.nshots, len(self.proto))
            elif len(self.nshots) != len(self.proto):
                raise ValueError("Length of nshots array does not match length of proto array")
        return self

    @property
    def vec_proto(self) -> np.ndarray:
        if self._vec_proto is None and self.proto is not None:
            self._vec_proto = meas_matrix(self.proto)
        return self._vec_proto

    @property
    def vec_nshots(self) -> np.ndarray:
        if self._vec_nshots is None and self.nshots is not None and self.proto is not None:
            n = [np.full((elem.shape[0],), n) for elem, n in zip(self.proto, self.nshots)]
            self._vec_nshots = np.concatenate(tuple(n))
        return self._vec_nshots

    @property
    def vec_clicks(self) -> np.ndarray:
        if self._vec_clicks is None and self.clicks is not None:
            self._vec_clicks = np.concatenate(tuple(self.clicks))
        return self._vec_clicks

    def stat(self) -> stat.Statistics:
        if type(self.stat_pkg) is str:
            if self.stat_pkg == "auto":
                if any(np.isinf(self.nshots)):
                    self.set_data(nshots=np.ones(self.nshots.shape))
                    self.stat_pkg = "asymptotic"
                else:
                    imat = np.eye(self.dim)
                    if all([elem.shape[0] == 1 for elem in self.proto]):
                        self.stat_pkg = "binomial"
                    elif self.entity is State and \
                            all([np.allclose(np.sum(elem, axis=0), imat) for elem in self.proto]):
                        self.stat_pkg = "polynomial"
                    elif self.entity is Process and \
                            all([np.allclose(part_trace(np.sum(elem, axis=0), [self.dim, self.dim], 0), imat) for elem in
                                 self.proto]):
                        self.stat_pkg = "polynomial"
                    else:
                        raise ValueError("Failed to determine statistics type. Please, specify stat_type manually.")
            self.stat_pkg = stat.BUILD_IN[self.stat_pkg]
        return self.stat_pkg

    def get_probs_dm(self, dm: np.ndarray, tol=0.0) -> np.ndarray:
        p = np.abs(self.vec_proto @ dm.reshape((-1,), order="F"))
        p[p < tol] = tol
        return p

    def get_probs_sq(self, sq: np.ndarray, tol=0.0) -> np.ndarray:
        return self.get_probs_dm(sq @ sq.conj().T, tol)

    def nkp(self, dm=None):
        n = self.vec_nshots
        k = self.vec_clicks
        if dm is None:
            return n, k
        else:
            return n, k, self.get_probs_dm(dm, 1e-15)

    # Sampling
    def simulate(self, dm: np.ndarray):
        clk = []
        for elem, n in zip(self.proto, self.nshots):
            probs = np.abs(meas_matrix(elem) @ dm.reshape((-1,), order="F"))
            clk.append(self.stat().sample(n, probs))
        self._vec_clicks = None
        self.set_data(clicks=clk)

    # Likelihood
    def logL_dm(self, dm: np.ndarray) -> float:
        n, k, p = self.nkp(dm)
        return self.stat().logL(n, k, p)

    def logL_sq(self, sq: np.ndarray) -> float:
        return self.logL_dm(sq @ sq.conj().T)

    def dlogL_sq(self, sq: np.ndarray) -> np.ndarray:
        n, k, p = self.nkp(sq @ sq.conj().T)
        b = self.stat().dlogL(n, k, p)
        bmat = self.vec_proto
        return 2 * np.reshape(bmat.conj().T @ b, (self.dim, -1), order="F") @ sq

    def logL_eq_mu(self) -> float:
        n, k = self.nkp()
        return self.stat().logL_mu(n, k)

    def logL_eq_jmat_dm(self, dm) -> np.ndarray:
        n, k, p = self.nkp(dm)
        b, b0 = self.stat().logL_jmat(n, k, p)
        bmat = self.vec_proto
        jmat = np.reshape(bmat.conj().T @ b, (self.dim, self.dim), order="F")
        if b0 != 0:
            jmat += b0 * np.eye(self.dim)
        return jmat

    # Chi-squared
    def chi2_dm(self, dm: np.ndarray):
        n, k, p = self.nkp(dm)
        return self.stat().chi2(n, k, p)

    def deg_f_rank(self, rank):
        nu = self.stat().deg_f(self.clicks)
        if self.entity is State:
            nu_dm = (2 * self.dim - rank) * rank - 1
        elif self.entity is Process:
            dim2 = self.dim ** 2
            nu_dm = (2 * dim2 - rank) * rank - dim2
        else:
            raise ValueError("Invalid entity")
        return nu - nu_dm


def nshots_divide(n, m, method="total_int"):
    if np.floor(n) != n:
        raise ValueError("Total shots number should be an integer")
    if np.isinf(n):
        return [np.inf] * m
    if method == "total":
        nshots = np.full((m,), n / m)
    elif method == "total_int":
        nshots = np.full((m,), np.floor(n / m))
        nshots[:int(n - np.sum(nshots))] += 1
    elif method == "equal":
        nshots = np.full((m,), n)
    else:
        raise ValueError("Invalid division method")
    return list(nshots)


def proto_measurement(ptype: str, dim=None, modifier="", num=None, nsub=1):
    ptype = ptype.lower()
    modifier = modifier.lower()
    if ptype == "mub":
        proto = [base2povm(base) for base in get_mubs(dim)]
    elif ptype == "tetra":
        proto = [base2povm(base) for base in get_tetra()]
    elif ptype == "random_bases":
        proto = [base2povm(rand_unitary(dim)) for _ in range(num)]
    elif ptype == "random_projectors":
        proto = [extend(State.random(dim, 1).dm, 3) for _ in range(num)]
    else:
        raise ValueError("Unknown measurement protocol type '{}'".format(ptype))

    if modifier[:8] == "operator":
        if modifier == "operator":
            proto = np.concatenate(tuple(proto), axis=0)
        else:
            idx = int(modifier[8:])
            proto = [extend(elem[idx, :, :], 3) for elem in proto]

    if type(proto) is not list:
        proto = [proto[j, :, :] for j in range(proto.shape[0])]

    if nsub > 1:
        proto_0 = proto.copy()
        for js in range(1, nsub):
            proto = [kron3d(p1, p2) for p1, p2 in product(proto, proto_0)]

    return proto


def get_mubs(dim: int):
    bases = [np.eye(dim)]
    if dim == 2:
        bases.append(np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        bases.append(np.array([[1, 1], [1j, -1j]]) / np.sqrt(2))
    elif dim == 3:
        w0 = 1 / np.sqrt(3)
        w1 = exp(1j * 2 * pi / 3) / np.sqrt(3)
        w2 = exp(1j * 4 * pi / 3) / np.sqrt(3)
        bases.append(np.array([[w0, w0, w0], [w0, w1, w2], [w0, w2, w1]]))
        bases.append(np.array([[w0, w0, w0], [w1, w2, w0], [w1, w0, w2]]))
        bases.append(np.array([[w0, w0, w0], [w2, w1, w0], [w2, w0, w1]]))
    elif dim == 4:
        w0 = 1 / 2
        w1 = 1j / 2
        w2 = -1 / 2
        w3 = -1j / 2
        bases.append(np.array([[w0, w0, w0, w0], [w0, w0, w2, w2], [w0, w2, w2, w0], [w0, w2, w0, w2]]))
        bases.append(np.array([[w0, w0, w0, w0], [w2, w2, w0, w0], [w3, w1, w1, w3], [w3, w1, w3, w1]]))
        bases.append(np.array([[w0, w0, w0, w0], [w3, w3, w1, w1], [w3, w1, w1, w3], [w2, w0, w2, w0]]))
        bases.append(np.array([[w0, w0, w0, w0], [w3, w3, w1, w1], [w2, w0, w2, w0], [w3, w1, w1, w3]]))
    elif dim == 5:
        w0 = 1 / np.sqrt(5)
        w1 = exp(1j * 2 * pi / 5) / np.sqrt(5)
        w2 = exp(1j * 4 * pi / 5) / np.sqrt(5)
        w3 = exp(1j * 6 * pi / 5) / np.sqrt(5)
        w4 = exp(1j * 8 * pi / 5) / np.sqrt(5)
        bases.append(np.array([[w0,w0,w0,w0,w0], [w0,w1,w2,w3,w4], [w0,w2,w4,w1,w3], [w0,w3,w1,w4,w2], [w0,w4,w3,w2,w1]]))
        bases.append(np.array([[w0,w0,w0,w0,w0], [w1,w2,w3,w4,w0], [w4,w1,w3,w0,w2], [w4,w2,w0,w3,w1], [w1,w0,w4,w3,w2]]))
        bases.append(np.array([[w0,w0,w0,w0,w0], [w2,w3,w4,w0,w1], [w3,w0,w2,w4,w1], [w3,w1,w4,w2,w0], [w2,w1,w0,w4,w3]]))
        bases.append(np.array([[w0,w0,w0,w0,w0], [w3,w4,w0,w1,w2], [w2,w4,w1,w3,w0], [w2,w0,w3,w1,w4], [w3,w2,w1,w0,w4]]))
        bases.append(np.array([[w0,w0,w0,w0,w0], [w4,w0,w1,w2,w3], [w1,w3,w0,w2,w4], [w1,w4,w2,w0,w3], [w4,w3,w2,w1,w0]]))
    else:
        raise ValueError("The only available MUBs are for dimensions 2, 3, 4, 5")
    return bases


def get_tetra():
    bases = []
    ap = np.sqrt((1 + 1 / np.sqrt(3)) / 2)
    am = np.sqrt((1 - 1 / np.sqrt(3)) / 2)
    w1 = exp(1j * 1 * pi / 4)
    w3 = exp(1j * 3 * pi / 4)
    w5 = exp(1j * 5 * pi / 4)
    w7 = exp(1j * 7 * pi / 4)
    bases.append(np.array([[ap, -am], [am * w1, ap * w1]]))
    bases.append(np.array([[am, -ap], [ap * w3, am * w3]]))
    bases.append(np.array([[ap, -am], [am * w5, ap * w5]]))
    bases.append(np.array([[am, -ap], [ap * w7, am * w7]]))
    return bases
