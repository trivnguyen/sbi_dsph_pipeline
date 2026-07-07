"""Prior box for the 8-parameter Zhao-Plummer dSph model.

Order: (dm_alpha, dm_beta, dm_gamma, dm_log_r_dm, dm_log_rho_0, df_beta0,
df_log_r_a), plus stellar_log_r_star as a conditioning dimension.
dm_log_r_dm/df_log_r_a are relative to stellar_log_r_star ("tilde" space);
to_physical()/to_tilde() convert between the two.

Fixed here as constants (not a config file) since it's unlikely to
change - matches npe/simulate_8params_process.py's PRIOR_MIN/PRIOR_MAX.
"""

import numpy as np
import torch

from .target import TargetData

PARAM_NAMES = [
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
    'dm_log_rho0', 'df_beta0', 'df_log_ra',
]
PRIOR_MIN = np.array([0.5, 2.0, -1.0, 0.0, 3.0, -0.499, -1.0])
PRIOR_MAX = np.array([3.0, 10.0, 2.0, 3.0, 10.0, 1.0, 3.0])

CONDITIONING_NAME = 'stellar_log_rstar'

# Parameters that are relative to the conditioning dimension (stellar_log_rstar)
# NOTE: although log_ra is relative to log_rstar, training is done in relative-space
# so we have to ignore it here to avoid double-counting the relative offset.
RELATIVE_PARAM_NAMES = ('dm_log_rdm', )
RELATIVE_INDICES = tuple(PARAM_NAMES.index(name) for name in RELATIVE_PARAM_NAMES)
CONDITIONING_INDEX = len(PARAM_NAMES)  # conditioning dim is always appended last

ALL_PARAM_NAMES = PARAM_NAMES + [CONDITIONING_NAME]


def conditioning_bounds(target: TargetData, n_sigma: float = 5.0) -> tuple[float, float]:
    """log10(r_half [kpc]) window, `n_sigma` wide, from the target's half-light radius.

    Args:
        target: Target snapshot providing rhalf_kpc and its uncertainty.
        n_sigma: Half-width of the window, in units of the (symmetrized)
            half-light-radius uncertainty.

    Returns:
        (log_min, log_max) bounds for the conditioning dimension.
    """
    err = 0.5 * (target.rhalf_kpc_em + target.rhalf_kpc_ep)
    return (
        float(np.log10(target.rhalf_kpc - n_sigma * err)),
        float(np.log10(target.rhalf_kpc + n_sigma * err)),
    )


def prior_box(target: TargetData, n_sigma: float = 5.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Full 8D prior box: fixed base params + conditioning bounds from `target`.

    Args:
        target: Target snapshot used to resolve the conditioning bounds.
        n_sigma: Passed to `conditioning_bounds`.

    Returns:
        (prior_min, prior_max) tensors, length 8 (7 base params + conditioning).
    """
    cond_min, cond_max = conditioning_bounds(target, n_sigma=n_sigma)
    prior_min = torch.tensor(np.append(PRIOR_MIN, cond_min), dtype=torch.float32)
    prior_max = torch.tensor(np.append(PRIOR_MAX, cond_max), dtype=torch.float32)
    return prior_min, prior_max


def sample_tilde(n_samples: int, target: TargetData, n_sigma: float = 5.0) -> np.ndarray:
    """Draw `n_samples` uniform samples from the prior box, in tilde space.

    Args:
        n_samples: Number of samples to draw.
        target: Target snapshot used to resolve the conditioning bounds.
        n_sigma: Passed to `conditioning_bounds`.

    Returns:
        (n_samples, 8) ndarray of tilde-space samples.
    """
    prior_min, prior_max = prior_box(target, n_sigma=n_sigma)
    dist = torch.distributions.Uniform(prior_min, prior_max)
    return dist.sample((n_samples,)).numpy()


def to_physical(theta_tilde: np.ndarray) -> np.ndarray:
    """Map tilde-space samples to physical units.

    `theta_tilde[:, i]` for `i in RELATIVE_INDICES` holds an *offset* from
    the conditioning value; physical = offset + conditioning.

    Args:
        theta_tilde: (N, 8) tilde-space samples.

    Returns:
        (N, 8) physical-unit samples.
    """
    theta_phys = np.asarray(theta_tilde, dtype=float).copy()
    cond = theta_phys[:, CONDITIONING_INDEX]
    for i in RELATIVE_INDICES:
        theta_phys[:, i] = theta_phys[:, i] + cond
    return theta_phys


def to_tilde(theta_phys: np.ndarray) -> np.ndarray:
    """Inverse of `to_physical`: map physical-unit samples back to tilde space.

    Args:
        theta_phys: (N, 8) physical-unit samples.

    Returns:
        (N, 8) tilde-space samples.
    """
    theta_tilde = np.asarray(theta_phys, dtype=float).copy()
    cond = theta_tilde[:, CONDITIONING_INDEX]
    for i in RELATIVE_INDICES:
        theta_tilde[:, i] = theta_tilde[:, i] - cond
    return theta_tilde


def default_norm_dict() -> dict:
    """Generic norm_dict for the random_init debug model - no real data needed.

    Returns:
        norm_dict with theta_loc/theta_scale (length 7, spanning the fixed
        prior box), cond_loc/cond_scale (length 1, unit scale), and
        x_loc/x_scale (length 3, unit scale).
    """
    theta_loc = (PRIOR_MAX + PRIOR_MIN) / 2
    theta_scale = (PRIOR_MAX - PRIOR_MIN) / 2
    return {
        'theta_loc': theta_loc.tolist(),
        'theta_scale': theta_scale.tolist(),
        'cond_loc': [0.0],
        'cond_scale': [1.0],
        'x_loc': [0.0, 0.0, 0.0],
        'x_scale': [1.0, 1.0, 1.0],
    }
