"""
Verbatim copy of sbi_twins/sbi_twins/sanders_evans.py (2026-09-22), so
the 14p Jeans spec does not import sbi_twins.

Sanders & Evans (2020) velocity kernels, parametrised by kurtosis.

GravSphere2 (Bañares-Hernández et al. 2025, github.com/dadams42/
GravSphere2) models every star's line-of-sight velocity with one of two
non-Gaussian PDFs from Sanders & Evans (2020): a Gaussian convolved
with a uniform kernel of half-width a (kurtosis below 3) or with a
Laplace kernel of scale a (kurtosis above 3), rescaled to the requested
variance and convolved analytically with the Gaussian measurement
error. It maps the model kurtosis to a with lookup tables covering
1.80 < kappa < 6.00 and ``np.interp`` clamps outside, so a model with
kappa > 6 is evaluated at kappa = 6. The tables, branch rule and
``log_pdf`` below follow ``gravsphere2.py`` (``asolveu``, ``asolvel``,
``lukpdf``, ``llkpdf``, ``lnpdfj``) so that a kurtosis twin drawn with
``sample`` has exactly GravSphere2's velocity likelihood.

Sampling is direct: with z standard normal,
    v = sigma (z + U(-a, a)) / sqrt(1 + a^2 / 3)       kappa < 3,
    v = sigma (z + Laplace(a)) / sqrt(2 a^2 + 1)      kappa >= 3.
"""

import numpy as np
from scipy.special import erfc, erfcx, log_ndtr

# Same tables as GravSphere2: a from 1e-6 to 100 on 1e4 points.
_A_GRID = np.linspace(1e-6, 100.0, 10 ** 4)
_KURT_UNIFORM = 3.0 - (2 * _A_GRID ** 4 / 15) * (1 + _A_GRID ** 2 / 3) ** -2
_KURT_LAPLACE = 3.0 + 12 * _A_GRID ** 4 * (2 * _A_GRID ** 2 + 1) ** -2
KURT_MIN = float(_KURT_UNIFORM.min())
KURT_MAX = float(_KURT_LAPLACE.max())

# Reason: at kappa = 3 both kernels degenerate to a Gaussian with a -> 0
# and the Laplace formulas divide by a; treat a band around 3 as Gaussian.
GAUSSIAN_BAND = 1e-4


def kernel_parameter(kurt: np.ndarray) -> np.ndarray:
    """
    Kernel half-width or scale a for the requested kurtosis.

    Args:
        kurt: Kurtosis <v^4> / <v^2>^2; clamped to [KURT_MIN, KURT_MAX]
            by the table lookup, as in GravSphere2.

    Returns:
        a for the uniform kernel where kurt < 3, for the Laplace kernel
        otherwise.
    """
    kurt = np.asarray(kurt, dtype=float)
    a_uniform = np.interp(kurt, _KURT_UNIFORM[::-1], _A_GRID[::-1])
    a_laplace = np.interp(kurt, _KURT_LAPLACE, _A_GRID)
    return np.where(kurt < 3.0, a_uniform, a_laplace)


def sample(
    sigma: np.ndarray, kurt: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """
    Draw one velocity per star from the kernel with the given moments.

    Args:
        sigma: Velocity dispersion per star [km/s].
        kurt: Kurtosis per star; values outside the table range are
            clamped, matching what the likelihood can express.
        rng: NumPy random generator.

    Returns:
        Velocities [km/s] with zero mean, dispersion sigma and kurtosis
        kurt (within the table range).
    """
    sigma = np.asarray(sigma, dtype=float)
    kurt = np.asarray(kurt, dtype=float)
    a = kernel_parameter(kurt)
    z = rng.standard_normal(sigma.shape)
    uniform = (z + rng.uniform(-a, a)) / np.sqrt(1 + a ** 2 / 3)
    laplace = (z + rng.laplace(0.0, a)) / np.sqrt(2 * a ** 2 + 1)
    return sigma * np.where(kurt < 3.0, uniform, laplace)


def _log_diff_exp(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
    """log(exp(log_a) - exp(log_b)) for log_a > log_b."""
    return log_a + np.log1p(-np.exp(log_b - log_a))


def _log_pdf_uniform(
    w: np.ndarray, werr: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """Log PDF of w = v / sigma for the uniform kernel (GravSphere2 lukpdf)."""
    b = np.sqrt(1 + a ** 2 / 3)
    t = np.sqrt(1 + b ** 2 * werr ** 2)
    bw = b * w
    upper = np.where(
        (bw + a) / t < 0,
        _log_diff_exp(log_ndtr((bw + a) / t), log_ndtr((bw - a) / t)),
        _log_diff_exp(log_ndtr(-(bw - a) / t), log_ndtr(-(bw + a) / t)))
    return np.log(0.5 * b / a) + upper


def _log_erfc_term(arg: np.ndarray, z: np.ndarray, log_pref: np.ndarray,
                   a: np.ndarray, bw_t: np.ndarray) -> np.ndarray:
    """
    log[b / (4a) exp(arg / 2a^2) erfc(z)] in the two stable forms of
    GravSphere2: plain for arg < 0, erfcx with the Gaussian factor for
    arg > 0 (where exp and erfc would overflow and underflow).
    """
    plain = log_pref + arg / (2 * a ** 2) + np.log(erfc(z))
    scaled = (0.5 * np.log(np.pi / 8) + log_pref + np.log(4.0)
              - 0.5 * bw_t ** 2 - 0.5 * np.log(2 * np.pi) + np.log(erfcx(z)))
    return np.where(arg < 0, plain, scaled)


def _log_pdf_laplace(
    w: np.ndarray, werr: np.ndarray, a: np.ndarray
) -> np.ndarray:
    """Log PDF of w = v / sigma for the Laplace kernel (GravSphere2 llkpdf)."""
    b = np.sqrt(2 * a ** 2 + 1)
    t = np.sqrt(1 + b ** 2 * werr ** 2)
    bw = b * w
    log_pref = np.log(b / (4 * a))
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        positive = _log_erfc_term(
            t ** 2 - 2 * a * bw, (t ** 2 - a * bw) / (np.sqrt(2) * t * a),
            log_pref, a, bw / t)
        negative = _log_erfc_term(
            t ** 2 + 2 * a * bw, (t ** 2 + a * bw) / (np.sqrt(2) * t * a),
            log_pref, a, bw / t)
    return np.logaddexp(positive, negative)


def log_pdf(
    v: np.ndarray, sigma: np.ndarray, kurt: np.ndarray, err: np.ndarray = 0.0
) -> np.ndarray:
    """
    GravSphere2's per-star log-likelihood of a line-of-sight velocity.

    Args:
        v: Observed velocities relative to the systemic velocity [km/s].
        sigma: Model dispersion at each star's radius [km/s].
        kurt: Model kurtosis at each star's radius (clamped to the
            table range).
        err: Gaussian measurement error per star [km/s].

    Returns:
        log p(v | sigma, kurt, err) per star.
    """
    v, sigma, kurt = np.broadcast_arrays(
        np.asarray(v, dtype=float), np.asarray(sigma, dtype=float),
        np.asarray(kurt, dtype=float))
    err = np.broadcast_to(np.asarray(err, dtype=float), v.shape)
    w, werr = v / sigma, err / sigma
    a = kernel_parameter(kurt)
    out = np.empty_like(w)
    gauss = np.abs(kurt - 3.0) < GAUSSIAN_BAND
    var = 1 + werr ** 2
    out[gauss] = (-0.5 * np.log(2 * np.pi * var) - 0.5 * w ** 2 / var)[gauss]
    uni = (kurt < 3.0) & ~gauss
    out[uni] = _log_pdf_uniform(w[uni], werr[uni], a[uni])
    lap = (kurt >= 3.0) & ~gauss
    out[lap] = _log_pdf_laplace(w[lap], werr[lap], a[lap])
    return out - np.log(sigma)
