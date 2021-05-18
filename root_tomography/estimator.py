import numpy as np
from typing import Union
from root_tomography.entity import State
from root_tomography.experiment import Experiment
from root_tomography.optimizer import FixedPoint, AutoRank
from root_tomography.tools import pinv, pval


def reconstruct_state(
        ex: Experiment,
        rank: Union[int, str] = "auto",
        significance_level: float = 0.05,
        get_stats: bool = False,
        init: Union[str, np.ndarray] = "pinv",
        reg_coeff: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = int(1e6),
        display: Union[int, bool] = False,
        return_rinfo: bool = False
):
    args = locals()
    dim = ex.dim
    if rank == "auto":
        optim = AutoRank()
        optim.set_options(display=display, significance_level=significance_level)
        rinfo, data_r = optim.run(dim, lambda r: rank_fun(args, r))
        state = rinfo["state"]
        del rinfo["state"]
        rinfo.update({"data_r": data_r})
        if return_rinfo:
            return state, rinfo
        else:
            return state
    elif rank == "full":
        rank = dim

    if rank < 1 or rank > dim:
        raise ValueError("Density matrix rank should be between 1 and Hilbert space dimension")

    if init == "pinv":
        p_est = ex.vec_clicks / ex.vec_nshots
        psi = pinv(ex.vec_proto, p_est, rank=rank).root
    else:
        psi = State.purify(init, rank)

    optim = FixedPoint()
    optim.set_options(display=display, max_iter=max_iter, tol=tol, reg_coeff=reg_coeff)
    mu_inv = 1 / ex.logL_eq_mu()
    f_val = lambda sq: mu_inv * ex.logL_eq_jmat_dm(sq @ sq.conj().T / np.trace(sq.conj().T @ sq)) @ sq
    psi, optim_info = optim.run(psi, f_val)
    state = State.from_root(psi / np.sqrt(np.trace(psi.conj().T @ psi)))

    rinfo = {
        "optimizer": optim,
        "iter": optim_info["iter"],
        "rank": rank
    }
    if get_stats:
        return_rinfo = True
        chi2 = ex.chi2_dm(state.dm)
        df = ex.deg_f_rank(rank)
        rinfo.update({"chi2": chi2, "df": df, "pval": pval(chi2, df)})

    if return_rinfo:
        return state, rinfo
    else:
        return state


def rank_fun(args, r):
    args["rank"] = r
    args["get_stats"] = True
    state, data = reconstruct_state(**args)
    data.update({"state": state})
    return data
