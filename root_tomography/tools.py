import numpy as np
import pickle
import os
from scipy import randn


# Reconstruction tools
def proto_check(proto, nshots=None, clicks=None):
    if type(proto) is not list:
        proto = [ proto[j,:,:] for j in range(proto.shape[0]) ]
    if type(nshots) is not None:
        if type(nshots) is str and nshots == "sum":
            lens = [ len(k) for k in clicks ]
            if np.any(lens == 1):
                raise ValueError("You must specify number of experiment repetitions for experiments with a single possible outcome")
            nshots = [ sum(k) for k in clicks ]
        if len(nshots) != len(proto):
            raise ValueError("Protocol array should have the same size as nshots array")
    return proto, nshots


def data_join(proto=None, nshots=None, clicks=None):
    if proto is not None:
        M = np.concatenate(tuple(proto), axis=0)
    else:
        M = []
    if nshots is not None:
        n = np.array([])
        for j in range(len(proto)): n = np.append(n, np.repeat(nshots[j], proto[j].shape[0]))
    else:
        n = []
    if clicks is not None:
        k = np.reshape(clicks, (-1,))
    else:
        k = []
    return M, n, k


def meas_matrix(M):
    return np.reshape(M, (M.shape[0], -1))


def projprob(p):
    if np.all( p >= 0):
        return p
    a = 0
    ind = np.argsort(p)
    for ji, i in enumerate(ind):
        irest = ind[ji:]
        nrest = len(irest)
        if p[i]+a/nrest >= 0:
            p[irest] += a/nrest
            break
        a += p[i]
        p[i] = 0
    return p


def fidelity(a, b):
    a = a/np.trace(a)
    b = b/np.trace(b)
    v, w, _ = np.linalg.svd(a)
    sqa = v.dot(np.diag(np.sqrt(w))).dot(v.conj().T)
    A = sqa.dot(b).dot(sqa)
    f = np.real(np.sum(np.sqrt(np.linalg.eigvals(A)))**2)
    if f > 1: f = 2-f  # fix computation inaccuracy
    return f


def purify(dm, r=None):
    w, v = np.linalg.eigh(dm)
    w = w[::-1]
    v = v[:,::-1]
    
    w = projprob(w)
    if type(r) is not None:
        w = w[0:r]
        v = v[:,0:r]
    return v*np.sqrt(w)


def pinv(M, p, r=None):
    d = M.shape[1]
    B = meas_matrix(M)
    dm = np.reshape(np.linalg.pinv(B).dot(p), (d,d), order="F")
    c = purify(dm, r)
    return c.dot(c.conj().T), c


def prttrace(dm, dim, sind):
    dms = 0
    Base1 = np.eye(dim[0])
    Base2 = np.eye(dim[1])
    for i in range(dim[sind]):
        Vec = np.kron(Base1[:,i],Base2) if sind == 0 else np.kron(Base1,Base2[:,i])
        dms = dms + Vec.conj().T.dot(dm).dot(Vec)
    return dms


def print_replace(text, nb=0, end=""):
    textsp = text + " "*max(0, nb-len(text))
    print("\r" + textsp, end=end, flush=True)
    return len(textsp)


def nshots_devide(n, m, method="total_int"):
    if method == "total":
        nshots = np.full((m,), n/m)
    elif method == "total_int":
        nshots = np.full((m,), np.floor(n/m))
        nshots[-1] = n - np.sum(nshots[0:-1])
    elif method == "equal":
        nshots = np.full((m,), n)
    else:
        raise ValueError("Unknown method")
    return nshots


# Simulation
def simulate(dm, proto, nshots=1, asymp=False):
    proto, nshots = proto_check(proto, nshots)
    m = len(proto)
    if type(nshots) is not list:
        nshots = nshots_devide(nshots, m, "equal")
    
    clicks = []
    for j in range(m):
        d = proto[j].shape[0]
        p = np.array([ np.real(np.trace(proto[j][i,:,:].dot(dm))) for i in range(d) ])
        if asymp:
            clicks.append(p*nshots[j])
        elif d == 1:
            clicks.append(np.random.poisson(p*nshots[j]))
        elif d == 2:
            k0 = np.random.binomial(nshots[j], p[0]/np.sum(p))
            k1 = nshots[j] - k0
            clicks.append(np.array([k0,k1]))
        else:
            clicks.append(np.random.multinomial(nshots[j], p/np.sum(p)))
    
    return clicks


# Protocol generator
def protocol(ptype: str, dim=None):
    if ptype.lower() == "mub":
        if type(dim) is not list:
            dim = [dim]
        bases = []
        for d in dim:
            if d not in [2, 3, 4, 8]:
                raise ValueError("Only MUB dimensions 2, 3, 4, 8 are supported")
            filename = os.path.dirname(__file__) + "/mubs.pickle"
            with open(filename, "rb") as handle:
                basis = pickle.load(handle)["mub{}".format(d)]
            bases = np.kron(bases, basis) if np.any(bases) else basis
        proto = [basis2povm(b) for b in bases]
    else:
        raise ValueError("Unknown protocol")
    return proto


def basis2povm(basis):
    d = basis.shape[0]
    m = basis.shape[1]
    povm = np.empty((m, d, d), dtype=complex)
    for j in range(m):
        povm[j, :, :] = np.outer(basis[:, j], basis[:, j].conj())
    return povm


# Generators
def gen_unitary(d):
    q, r = np.linalg.qr(randn(d, d) + 1j * randn(d, d))
    r = np.diag(r)
    return q * (r / abs(r))


def gen_state(stype, d, rank=None):
    if stype == "haar_vec":
        x = randn(d) + 1j * randn(d)
        return x / np.linalg.norm(x)
    elif stype == "haar_dm":
        psi = gen_state("haar_vec", d*rank)
        psi = np.reshape(psi, (d, rank), order="F")
        return psi.dot(psi.conj().T)
    elif stype == "bures_dm":
        G = randn(d, d) + 1j * randn(d, d)
        U = gen_unitary(d)
        A = (np.eye(d) + U).dot(G)
        dm = A.dot(A.conj().T)
        return dm / np.trace(dm)
    else:
        raise ValueError("Unknown state type: {}".format(stype))
