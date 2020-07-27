import numpy as np
import root_tomography.tools as tools
from root_tomography.adequacy import significance


def dm_reconstruct(clicks,
                   proto,
                   nshots="sum",
                   rank="auto",
                   significance_level=0.05,
                   normalize=True,
                   init="pinv",
                   pinv_only=False,
                   alpha=0.5,
                   tol=1e-8,
                   max_iter=1e6,
                   display=False
                   ):
    
    kwargs = locals()
    proto, nshots = tools.proto_check(proto, nshots, clicks)
    d = proto[0].shape[1]
    if rank == "auto":
        if display:
            print("=== Automatic rank estimation ===")
        pvalRed = False
        dm_r, rinfo_r = [], []
        for i in range(d):
            r = i+1
            if display:
                print("Try rank {}".format(r))
            kwargs["rank"] = r
            dm, rinfo = dm_reconstruct(**kwargs)
            dm_r.append(dm)
            rinfo_r.append(rinfo)
            if np.isnan(rinfo["pval"]) or rinfo["pval"] >= significance_level: break
            elif r > 1 and rinfo["pval"] < rinfo_r[i-1]["pval"]:
                pvalRed = True
                dm, rinfo = dm_r[i-1], rinfo_r[i-1]
                break
        if display:
            if rinfo["pval"] >= significance_level:
                print("Rank {:d} is statistically significant at significance level {}. Procedure terminated.".format(rinfo["rank"], significance_level))
            elif pvalRed:
                print("P-value is maximal ({}) for rank {:d}. Procedure terminated.".format(round(rinfo["pval"],2), rinfo["rank"]))
            else:
                print("Failed to determine optimal rank. Maximal rank {:d} is taken.".format(rinfo["rank"]))

        rinfo.update({"dm_r": dm_r[0:rinfo["rank"]], "info_r": rinfo_r[0:rinfo["rank"]]})
        return dm, rinfo
    elif rank == "full":
        rank = d
    
    if rank < 1 or rank > proto[0].shape[1]:
        raise ValueError("Density matrix rank should be an integer between 1 and Hilbert space dimension")
    
    M, n, k = tools.data_join(proto, nshots, clicks)
    if (type(init) is str and init == "pinv") or pinv_only:
        if display:
            h = tools.print_replace("Pseudo-inversion...")
        _, c = tools.pinv(M, k/n, rank)
    if type(init) == np.ndarray:
        c = tools.purify(init, rank)
    
    niter = 0
    if not pinv_only:
        dispfreq = 1
        d = c.shape[0]
        B = tools.meas_matrix(M)
        Ir = np.linalg.inv(np.reshape(B.conj().T.dot(n), (d, d), order="F"))
        for i in range(int(max_iter)):
            cp = c
            
            dmk = c.dot(c.conj().T)
            p = np.real(B.dot(np.reshape(dmk, (-1,), order="F")))
            p[np.where(p<1e-15)] = 1e-15
            J = np.reshape(B.conj().T.dot(k/p), (d, d), order="F")
            cn = Ir.dot(J).dot(c)
            
            c = (1-alpha)*cn + alpha*cp
            dc = np.linalg.norm(cp-c)
            stop_iter = dc < tol
            if display and (np.mod(i,dispfreq) == 0 or i == 0 or stop_iter):
                h = tools.print_replace("Iteration {:d}      Difference {:.4e}".format(i + 1, dc), h)
            
            if stop_iter: break
        niter = i+1
    if display:
        h = tools.print_replace("", end="\n")
    
    dm = c.dot(c.conj().T)
    chi2, pval, df, n_obs, n_exp = significance(dm, clicks, proto, nshots, rank=rank)
    if normalize: dm = dm/np.trace(dm)
    
    rinfo = {"niter": niter, "rank": rank, "pval": pval, "chi2": chi2, "df": df, "n_observed": n_obs, "n_expected": n_exp}
    return dm, rinfo