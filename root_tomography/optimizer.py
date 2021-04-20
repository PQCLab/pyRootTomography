import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from root_tomography.tools import uprint


class Optimizer(ABC):
    display = True

    def set_options(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    @abstractmethod
    def run(self, **kwargs):
        pass


class FixedPoint(Optimizer):
    max_iter = int(1e6)
    tol = 1e-8
    reg_coeff = 0.5

    def run(self, x0: np.ndarray, f_val: Callable):
        nb = 0
        if self.display:
            print("Optimization: fixed point iteration method")
            nb = uprint("Starting optimization")
        x = x0
        iter = 0
        for iter in range(self.max_iter):
            xp = x
            x = (1 - self.reg_coeff) * f_val(xp) + self.reg_coeff * xp
            dx = np.linalg.norm(xp - x, ord="fro")
            stop_iter = dx < self.tol
            if self.display and (np.mod(iter, self.display) == 0 or stop_iter):
                nb = uprint("Iteration {} \t\t Delta {:.2e}".format(iter + 1, dx), nb)
            if stop_iter:
                break
        if self.display:
            uprint("", end="\n")
        info = {"iter": iter + 1}
        return x, info


class AutoRank(Optimizer):
    significance_level = 0.05

    def run(self, r_max: int, f_data: Callable):
        if self.display:
            print("=== Automatic rank estimation ===")
        r = 0
        pval_red = False
        info = [None] * r_max
        for idx in range(r_max):
            r = idx + 1
            if self.display:
                print(f"=> Try rank {r:d}")
            info[idx] = f_data(r)
            if np.isnan(info[idx]["pval"]) or info[idx]["pval"] > self.significance_level:
                break
            if idx > 0 and info[idx]["pval"] < info[idx - 1]["pval"]:
                pval_red = True
                r = r - 1
                break
        if self.display:
            if info[r - 1]["pval"] > self.significance_level:
                print(f"=> Rank {r:d} is statistically significant at significance level {self.significance_level:.5f}. Procedure terminated.")
            elif pval_red:
                print(f"=> P-value is maximal ({info[r-1]['pval']:.5f}) for rank {r:d}. Procedure terminated.")
            else:
                print(f"=> Failed to determine optimal rank. Maximal rank {r_max:d} is taken")
        return info[r - 1], info
