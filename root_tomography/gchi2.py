import numpy as np
from scipy.stats.distributions import chi2
from scipy.special import iv as besseli
from scipy import fft


def pdf(d, x=None, tol=1e-4, n_bins=1e4):
    n_bins = int(n_bins)

    x0 = x
    xmax = chi2.ppf(1 - tol, 1) * sum(d)
    dx = xmax / n_bins
    x = np.array(range(n_bins - 20)) * dx
    nu = len(d)

    if nu < 3:
        # Exact solution for 1 or 2 degrees of freedom
        if x0 is not None:
            x = x0
        if nu == 1:
            p = chi2.pdf(x / d[0], 1) / d[0]
        else:
            a = d[0]
            b = d[1]
            p = 1 / (2 * np.sqrt(a * b)) * np.exp(-(a + b) * x / (4 * a * b)) * besseli(0, (a - b) * x / (4 * a * b))
    else:
        # Numerical solution for arbitrary degrees of freedom
        du = 2 * np.pi / xmax
        u = np.array(range(n_bins)) * du
        ug, dg = np.meshgrid(u, d)
        phi = 1 / np.sqrt(1 - 2 * 1j * ug * dg)
        phi = np.prod(phi, axis=0)
        p = (du / np.pi) * np.real(fft.fft(phi))
        p = p[:(n_bins - 20)] - min(p)
        if x0 is not None:
            p = np.interp(x0, x, p)
            x = x0

    if x0 is None:
        return p, x
    else:
        return p


def cdf(d, x=None, tol=1e-4, n_bins=1e4):
    x0 = x
    p, x = pdf(d, tol=tol, n_bins=n_bins)
    f = np.cumsum(p) * (x[1] - x[0])
    if x0 is None:
        return f, x
    else:
        f = np.interp(x0, x, f)
        return f


def ppf(d, p, tol=1e-4, n_bins=1e4):
    f, x = cdf(d, tol=tol, n_bins=n_bins)
    idx0 = np.where(f > 1e-10)[0][0]
    idx1 = np.where(f < 1 - 1e-10)[0][-1]
    x = x[idx0:(idx1+1)]
    f = f[idx0:(idx1 + 1)]
    f, idx = np.unique(f, return_index=True)
    x = x[idx]
    return np.interp(p, f, x)