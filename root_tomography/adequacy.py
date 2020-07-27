import numpy as np
from scipy.special import gammaincc
import root_tomography.tools as tools


def significance(dm,
                 clicks,
                 proto,
                 nshots="sum",
                 from_clicks=True,
                 rank="dm",
                 is_process=False,
                 normalize_dm=True,
                 meas_df="proto"
                 ):
    
    # getting expected and observed counts
    proto, nshots = tools.proto_check(proto, nshots, clicks)
    M, n, n_observed = tools.data_join(proto, nshots, clicks)
    n_expected = np.real(tools.meas_matrix(M).dot(np.reshape(dm, (-1,), order="F")))*n
    if normalize_dm:
        n_expected = n_expected / np.sum(n_expected) * np.sum(n_observed)
    
    # getting degrees of freedom
    d = dm.shape[0]
    if meas_df == "proto":
        if is_process:
            ds = np.sqrt(d)
            nPovm = sum([ np.linalg.norm(tools.prttrace(np.sum(povm, axis=0), [ds,ds], 0)-np.eye(ds)) < 1e-5 for povm in proto ])
        else:
            nPovm = sum([ np.linalg.norm(np.sum(povm, axis=0)-np.eye(d)) < 1e-5 for povm in proto ])
        df = len(n_observed) - nPovm - (normalize_dm and (nPovm < len(proto)))
    else:
        df = meas_df
    
    if from_clicks:
        if rank == "dm":
            rank = np.rank(dm)
        nuP = (2*d-rank)*rank - (d if is_process else 1)
        df = df -  nuP
    
    # chi-squared and p-value
    zeros = np.where(n_expected == 0)[0]
    if len(zeros) and np.any(np.nonzero(n_observed[zeros])):
        chi2 = np.Inf
        pval = 0
    else:
        ne, no = np.delete(n_expected, zeros), np.delete(n_observed, zeros)
        chi2 = np.sum((ne-no)**2/ne)
        pval = gammaincc(df/2, chi2/2) if df > 0 else np.nan
    
    return chi2, pval, df, n_observed, n_expected
