import numpy as np
from typing import Union
from root_tomography.entity import State
from root_tomography.experiment import Experiment
from root_tomography.optimizer import FixedPoint, AutoRank
from root_tomography.tools import pinv, pval


def reconstruct_state(
        dim: int,
        clicks: Union[list, np.ndarray],
        proto: Union[list, np.ndarray],
        nshots: Union[int, list] = "sum",
        stat_type: str = "auto",
        rank: Union[int, str] = "auto",
        significance_level: float = 0.05,
        get_stats: bool = False,
        init: Union[str, np.ndarray] = "pinv",
        reg_coeff: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = int(1e6),
        display: Union[int, bool] = False
):
    args = locals()
    if rank == "auto":
        optim = AutoRank()
        optim.set_options(display=display, significance_level=significance_level)
        rinfo, data_r = optim.run(dim, lambda r: rank_fun(args, r))
        state = rinfo["state"]
        del rinfo["state"]
        rinfo.update({"data_r": data_r})
        return state, rinfo
    elif rank == "full":
        rank = dim
    if rank < 1 or rank > dim:
        raise ValueError("Density matrix rank should be between 1 and Hilbert space dimension")

    if nshots == "sum":
        nshots = [sum(k) for k in clicks]

    ex = Experiment(dim, State, stat_type)
    ex.set_data(proto=proto, clicks=clicks, nshots=nshots)

    if init == "pinv":
        p_est = ex.vec_clicks / ex.vec_nshots
        psi_0 = pinv(ex.proto, p_est, rank=rank).root
    else:
        psi_0 = State.purify(init, rank)

    optim = FixedPoint()
    optim.set_options(display=display, max_iter=max_iter, tol=tol, reg_coeff=reg_coeff)
    imat = np.reshape(2 * ex.vec_proto.conj().T @ ex.vec_nshots, (dim, dim), order="F")
    imat_inv = np.linalg.inv(imat)

    if ex.stat_type == "poly":
        f_val = lambda sq: imat_inv @ ex.get_dlogL_sq(sq)
    elif ex.stat_type == "poiss":
        f_val = lambda sq: imat_inv @ ex.get_dlogL_sq(sq) + sq
    else:
        raise ValueError("Invalid statistics type")

    sq, optim_info = optim.run(psi_0, f_val)
    state = State.from_root(sq)

    rinfo = {
        "optimizer": optim,
        "iter": optim_info["iter"],
        "rank": rank,
        "experiment": ex
    }
    if get_stats:
        chi2 = ex.get_chi2_dm(state.dm)
        df = ex.get_df(rank)
        rinfo.update({"chi2": chi2, "df": df, "pval": pval(chi2, df)})

    return state, rinfo


def rank_fun(args, r):
    args["rank"] = r
    args["get_stats"] = True
    state, data = reconstruct_state(**args)
    data.update({"state": state})
    return data
