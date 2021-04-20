import numpy as np
from cmath import exp, pi
from itertools import product
from root_tomography.entity import State, Process, part_trace
from root_tomography.tools import kron3d, extend, base2povm, meas_matrix
rng = np.random.default_rng()


class Experiment:
    dim = None
    entity = None
    stat_type = None
    proto = None
    nshots = None
    clicks = None
    _vec_proto = None
    _vec_nshots = None
    _vec_clicks = None

    def __init__(self, dim: int, entity, stat_type="auto"):
        self.dim = dim
        self.entity = entity
        if self.entity.__name__ not in ["State", "Process"]:
            raise ValueError(
                "Unknown entity: '{}'\n Only State, Process are available".format(entity.__name__))
        self.stat_type = stat_type.lower()
        if self.stat_type not in ["poly", "poiss", "asymp", "auto"]:
            raise ValueError(
                "Unknown statistics type: '{}'\n Only 'poly', 'poiss', 'asymp', 'auto' are available".format(stat_type))

    def set_data(self, proto=None, nshots=None, clicks=None):
        if proto is not None:
            if self.entity is Process:
                pass

            if type(proto) is not list:
                proto = [proto[j, :, :] for j in range(proto.shape[0])]
            proto = [extend(elem, 3) for elem in proto]

            if self.stat_type == "auto":
                imat = np.eye(self.dim)
                if all([elem.shape[0] == 1 for elem in proto]):
                    self.stat_type = "poiss"
                elif self.entity is State and \
                        all([np.allclose(np.sum(elem, axis=0), imat) for elem in proto]):
                    self.stat_type = "poly"
                elif self.entity is Process and \
                        all([np.allclose(part_trace(np.sum(elem, axis=0), [self.dim, self.dim], 0), imat) for elem in proto]):
                    self.stat_type = "poly"
                else:
                    raise ValueError("Failed to determine statistics type. Please, specify stat_type manually.")
            self.proto = proto
            self._vec_proto = None

        if clicks is not None:
            if type(clicks) is not list:
                clicks = [clicks[:, j] for j in range(clicks.shape[1])]
            self.clicks = clicks
            self._vec_clicks = None

        if nshots is not None:
            if self.stat_type == "auto" and np.any(nshots is np.inf):
                nshots = [1] * len(nshots)
                self.stat_type = "asymp"
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

    def get_probs_dm(self, dm: np.ndarray, tol=0.0) -> np.ndarray:
        p = np.abs(self.vec_proto @ dm.reshape((-1,), order="F"))
        p[p < tol] = tol
        return p

    def get_probs_sq(self, sq: np.ndarray, tol=0.0) -> np.ndarray:
        return self.get_probs_dm(sq @ sq.conj().T, tol)

    # Sampling
    def simulate(self, dm: np.ndarray) -> list:
        clicks = []
        for elem, n in zip(self.proto, self.nshots):
            probs = np.abs(meas_matrix(elem) @ dm.reshape((-1,), order="F"))
            clicks.append(self.sample(probs, n))
        return clicks

    def sample(self, p: np.ndarray, n: int) -> np.ndarray:
        if self.stat_type == "poly":
            if abs(sum(p) - 1) > 1e-8:
                raise ValueError("For simulating polynomial statistics probabilities in each measurement should sum to unity")
            p = p / sum(p)
            if len(p) == 2:
                k = np.array([0, 0])
                k[0] = rng.binomial(n, p[0])
                k[1] = n - k[0]
            else:
                k = rng.multinomial(n, p)
        elif self.stat_type == "poiss":
            k = rng.poisson(n * p)
        elif self.stat_type == "asymp":
            k = n * p
        else:
            raise ValueError("Invalid statistics type")
        return k

    # Likelihood
    def get_logL_dm(self, dm: np.ndarray):
        p = self.get_probs_dm(dm, 1e-15)
        k = self.vec_clicks
        if self.stat_type == "poly":
            return sum(k * np.log(p))
        elif self.stat_type == "poiss":
            lam = self.vec_nshots * p
            return k * np.log(lam) - lam
        else:
            raise ValueError("Invalid statistics type")

    def get_logL_sq(self, sq: np.ndarray):
        return self.get_logL_dm(sq @ sq.conj().T)

    def get_dlogL_sq(self, sq: np.ndarray):
        p = self.get_probs_sq(sq, 1e-12)
        k = self.vec_clicks
        bmat = self.vec_proto
        a = k / p
        if self.stat_type == "poly":
            jmat = np.reshape(bmat.conj().T @ a, (sq.shape[0], -1), order="F")
            return 2 * jmat @ sq
        elif self.stat_type == "poiss":
            jmat = np.reshape(bmat.conj().T @ (a - self.vec_nshots), (sq.shape[0], -1), order="F")
            return 2 * jmat @ sq
        else:
            raise ValueError("Invalid statistics type")

    # Chi-squared
    def get_chi2_dm(self, dm: np.ndarray):
        n_expected = self.get_probs_dm(dm) * self.vec_nshots
        n_observed = self.vec_clicks
        return sum((n_expected-n_observed) ** 2 / n_expected)

    def get_df(self, rank):
        df = len(self.vec_clicks)
        if self.stat_type == "poly":
            df -= len(self.clicks)
        if self.entity is State:
            nu = (2 * self.dim - rank) * rank - 1
        elif self.entity is Process:
            dim2 = self.dim ** 2
            nu = (2 * dim2 - rank) * rank - dim2
        else:
            raise ValueError("Invalid entity")
        df -= nu
        return df


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


def proto_measurement(ptype: str, dim=None, modifier="", nsub=1):
    ptype = ptype.lower()
    modifier = modifier.lower()
    if ptype == "mub":
        bases = get_mubs(dim)
        proto = [base2povm(base) for base in bases]
        if modifier == "operator":
            proto = np.concatenate(tuple(proto), axis=0)
    elif ptype == "tetra":
        bases = get_tetra()
        proto = [base2povm(base) for base in bases]
        if modifier == "operator":
            proto = np.concatenate(tuple(proto), axis=0)
        elif modifier == "operator+":
            proto = [extend(elem[0, :, :], 3) for elem in proto]
            proto = np.concatenate(tuple(proto), axis=0)
        elif modifier == "operator-":
            proto = [extend(elem[1, :, :], 3) for elem in proto]
            proto = np.concatenate(tuple(proto), axis=0)
    else:
        raise ValueError("Unknown measurement protocol type '{}'".format(ptype))

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
