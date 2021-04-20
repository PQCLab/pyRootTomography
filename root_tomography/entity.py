import numpy as np


class State:

    dim = None
    _dm = None
    _root = None
    _rank = None

    @classmethod
    def from_root(cls, root: np.ndarray):
        state = cls()
        state.dim = root.shape[0]
        state._root = root
        state._rank = root.shape[1]
        return state

    @classmethod
    def from_dm(cls, dm: np.ndarray):
        state = cls()
        state.dim = dm.shape[0]
        state._dm = dm
        return state

    @classmethod
    def random(cls, dim: int, rank=None):
        if not rank:
            rank = dim
        root = np.random.normal(size=(dim, rank)) + 1j * np.random.normal(size=(dim, rank))
        root = root / np.linalg.norm(root, "fro")
        state = cls.from_root(root)
        state._rank = rank
        return state

    @property
    def root(self) -> np.ndarray:
        if self._root is None and self._dm is not None:
            self._root = self.purify(self._dm, self.rank)
        return self._root

    @property
    def dm(self) -> np.ndarray:
        if self._dm is None and self._root is not None:
            self._dm = self._root @ self._root.conj().T
        return self._dm

    @property
    def rank(self):
        if self._rank is None and self._dm is not None:
            self._rank = np.linalg.matrix_rank(self._dm, hermitian=True)
        return self._rank

    @staticmethod
    def purify(a: np.ndarray, rank=None) -> np.ndarray:
        p, u = np.linalg.eigh(a)
        c = u * np.sqrt(project_to_simplex(p))
        c = c[:, ::-1]
        if rank:
            c = c[:, :rank]
        return c

    @classmethod
    def fidelity(cls, s1, s2):
        if type(s1) is not cls or type(s2) is not cls:
            raise ValueError("Inputs must be of type '{}'".format(cls.__name__))

        dm1 = s1.dm
        dm1 = dm1 / np.trace(dm1)
        dm2 = s2.dm
        dm2 = dm2 / np.trace(dm2)
        if s1.rank == 1 or s2.rank == 1:
            return np.abs(np.trace(dm1 @ dm2))

        dm1sq = cls.purify(dm1)
        lam = project_to_simplex(np.linalg.eigvalsh(dm1sq.conj().T @ dm2 @ dm1sq))
        f = np.abs(np.sum(np.sqrt(lam)) ** 2)
        return f

    def __eq__(self, other):
        return np.allclose(self.dm, other.dm)


def part_trace(dm, dim, sind):
    dms = 0
    base1 = np.eye(dim[0])
    base2 = np.eye(dim[1])
    for j in range(dim[sind]):
        if sind == 0:
            vec = np.kron(base1[:, j], base2)
        else:
            vec = np.kron(base1, base2[:, j])
        dms = dms + vec.conj().T @ dm @ vec
    return dms


def project_to_simplex(p: np.ndarray, maintain_sum=True):
    d = len(p)
    ps = sum(p) if maintain_sum else 1
    srt_idx = p.argsort()[::-1]
    p = p[srt_idx]
    mu = (np.cumsum(p) - ps) / np.arange(1, d + 1)
    idx = np.where(p - mu > 0)[0][-1]
    p = p - mu[idx]
    p[p < 0] = 0
    p[srt_idx] = p.copy()
    return p


class Process(State):

    dim2 = None
    _kraus = None
    _is_tp = True

    @classmethod
    def from_kraus(cls, kraus: np.ndarray):
        process = cls()
        process.dim = kraus[0].shape[0]
        process.dim2 = process.dim ** 2
        process._kraus = kraus
        process._rank = len(kraus)
        return process

    @classmethod
    def from_chi(cls, chi: np.ndarray):
        process = cls()
        process.dim2 = chi.shape[0]
        process.dim = int(np.sqrt(process.dim2))
        process._dm = chi
        return process

    @classmethod
    def from_root(cls, root: np.ndarray):
        process = super().from_root(root)
        process.dim2 = process.dim
        process.dim = int(np.sqrt(process.dim2))
        return process

    @classmethod
    def random(cls, dim: int, rank=1, trace_preserving=True):
        if trace_preserving:
            u = rand_unitary(dim * rank)
            u = u[:, :dim]
            kraus = u.reshape(rank, dim, dim)
            process = cls.from_kraus(kraus)
        else:
            process = super().random(dim, rank)
        process._is_tp = trace_preserving
        return process

    @property
    def chi(self):
        if self._dm is None:
            e = self.root
            self._dm = e @ e.conj().T
        return self._dm

    @property
    def kraus(self):
        if self._kraus is None:
            e = self.root
            self._kraus = e.reshape(self.dim, self.dim, self.rank).transpose(2, 1, 0)
        return self._kraus

    @property
    def root(self):
        if self._root is None:
            if self._kraus is not None:
                self._root = self._kraus.transpose(2, 1, 0).reshape(self.dim2, self.rank)
            else:
                self._root = super().root
        return self._root


def rand_unitary(dim: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.normal(size=(dim, dim)) + 1j*np.random.normal(size=(dim, dim)))
    r = np.diag(r)
    return q @ np.diag(r / np.abs(r))
