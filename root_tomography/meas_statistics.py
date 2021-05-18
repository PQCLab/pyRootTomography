from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
rng = np.random.default_rng()


class Statistics(ABC):
    @staticmethod
    @abstractmethod
    def sample(n: int, p: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def logL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def dlogL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def logL_mu(n: np.ndarray, k: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def logL_jmat(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
        pass

    @staticmethod
    @abstractmethod
    def chi2(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def deg_f(clicks: np.ndarray) -> int:
        pass

    @staticmethod
    @abstractmethod
    def fisher_information(n: np.ndarray, p: np.ndarray) -> np.ndarray:
        pass


class Polynomial(Statistics):
    @staticmethod
    def sample(n: int, p: np.ndarray) -> np.ndarray:
        if abs(sum(p) - 1) > 1e-8:
            raise ValueError(
                "For simulating polynomial statistics probabilities in each measurement should sum to unity")
        p = p / sum(p)
        if len(p) == 2:
            k = np.array([0, 0])
            k[0] = rng.binomial(n, p[0])
            k[1] = n - k[0]
        else:
            k = rng.multinomial(n, p)
        return k

    @staticmethod
    def logL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        return float(np.sum(k * np.log(p)))

    @staticmethod
    def dlogL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        return k / p

    @staticmethod
    def logL_mu(n: np.ndarray, k: np.ndarray) -> float:
        return float(np.sum(k))

    @staticmethod
    def logL_jmat(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
        return k / p, 0

    @staticmethod
    def chi2(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        ne = p * n
        no = k
        return float(np.sum((ne - no)**2 / ne))

    @staticmethod
    def deg_f(clicks: np.ndarray) -> int:
        return sum([len(k) - 1 for k in clicks])

    @staticmethod
    def fisher_information(n: np.ndarray, p: np.ndarray) -> np.ndarray:
        return n / p


class Binomial(Statistics):
    @staticmethod
    def sample(n: int, p: np.ndarray) -> np.ndarray:
        if len(p) > 1:
            raise ValueError(
                "Only a single result per measurement is allowed for binomial statistics simulation")
        return rng.binomial(n, p)

    @staticmethod
    def logL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        return float(np.sum(k * np.log(p) + (n - k) * np.log(1 - p)))

    @staticmethod
    def dlogL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        return k / p - (n - k) / (1 - p)

    @staticmethod
    def logL_mu(n: np.ndarray, k: np.ndarray) -> float:
        return float(np.sum(k))

    @staticmethod
    def logL_jmat(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
        b = k / p - (n - k) / (1 - p)
        b0 = float(np.sum((n - k) * p / (1 - p)))
        return b, b0

    @staticmethod
    def chi2(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        ne = np.concatenate((p * n, (1 - p) * n))
        no = np.concatenate((k, n - k))
        return float(np.sum((ne - no)**2 / ne))

    @staticmethod
    def deg_f(clicks: np.ndarray) -> int:
        return len(clicks)

    @staticmethod
    def fisher_information(n: np.ndarray, p: np.ndarray) -> np.ndarray:
        return n / p / (1 - p)


class Poisson(Statistics):
    @staticmethod
    def sample(n: int, p: np.ndarray) -> np.ndarray:
        if len(p) > 1:
            raise ValueError(
                "Only a single result per measurement is allowed for poisson statistics simulation")
        return rng.poisson(n * p)

    @staticmethod
    def logL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        lam = n * p
        return float(np.sum(k * np.log(lam) - lam))

    @staticmethod
    def dlogL(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        return k / p - n

    @staticmethod
    def logL_mu(n: np.ndarray, k: np.ndarray) -> float:
        return float(np.sum(k))

    @staticmethod
    def logL_jmat(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
        b = k / p - n
        b0 = float(np.sum(n * p))
        return b, b0

    @staticmethod
    def chi2(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> float:
        ne = p * n
        no = k
        return float(np.sum((ne - no) ** 2 / ne))

    @staticmethod
    def deg_f(clicks: np.ndarray) -> int:
        return len(clicks)

    @staticmethod
    def fisher_information(n: np.ndarray, p: np.ndarray) -> np.ndarray:
        return n / p


class PoissonUnity(Poisson):
    @staticmethod
    def logL_jmat(n: np.ndarray, k: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, float]:
        b = k / p
        b0 = 0
        return b, b0


class Asymptotic(Poisson):
    @staticmethod
    def sample(n: int, p: np.ndarray) -> np.ndarray:
        return p * n


BUILD_IN = {
    "polynomial": Polynomial,
    "binomial": Binomial,
    "poisson": Poisson,
    "poisson_unity": PoissonUnity,
    "asymptotic": Asymptotic
}
