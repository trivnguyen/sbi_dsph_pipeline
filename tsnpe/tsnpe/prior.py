"""Prior box for the 8-parameter Zhao-Plummer dSph model.

Order: (dm_alpha, dm_beta, dm_gamma, dm_log_rdm, dm_log_rho0, df_beta0,
df_log_ra), plus stellar_log_rstar as a conditioning dimension.

Radius units
------------
The prior box always draws dm_log_rdm as log10(r_dm / r_star) -- an
offset from the conditioning value -- because that is the only space in
which the box is a box: r_dm's kpc bounds slide with each row's own
r_star draw. What differs between training sets is the units the *model*
predicts and tsnpe/sims.py consumes, selected by `RADIUS_UNITS`:

  'kpc'   (priorA / 8p_ZhaoPlumCOM, legacy) - the model predicts
          log10(r_dm / kpc), so to_kpc() adds the conditioning value on
          the way out and to_rstar_units() subtracts it on the way back.

  'rstar' (priorB / 8p_ZhaoPlumCOM_v3) - the model predicts
          log10(r_dm / r_star), the same units the box is defined in.
          Nothing converts: RSTAR_SCALED_PARAM_NAMES is empty and both
          functions become the identity. Switching to priorB is this one
          constant, plus retraining round 0 on the matching dataset.

df_log_ra is in r_star units under *both* conventions -- tsnpe/sims.py
multiplies it by r_star itself -- so it never takes part in either, and
must not be added to RSTAR_SCALED_PARAM_NAMES or the offset gets applied
twice.

Bounds are constants rather than a config file since they change only
with the training set, but they must match the `prior_min`/`prior_max`
recorded in that set's config.<n>.json.
"""

import numpy as np
import torch

from .target import TargetData

PARAM_NAMES = [
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
    'dm_log_rho0', 'df_beta0', 'df_log_ra',
]

CONDITIONING_NAME = 'stellar_log_rstar'
CONDITIONING_INDEX = len(PARAM_NAMES)  # conditioning dim is always appended last
ALL_PARAM_NAMES = PARAM_NAMES + [CONDITIONING_NAME]

# Which units the round-0 model predicts its radii in. See the module
# docstring; must match the dataset that model was trained on.
RADIUS_UNITS = 'kpc'

_PRIOR_BOUNDS = {
    # priorA / 8p_ZhaoPlumCOM. dm_beta's lower bound was 2.0 here while
    # that dataset's recorded prior_min has 1.0, so the proposal could
    # never revisit beta < 2 even though round 0 was trained on it;
    # widened to 1.0 to match the training set.
    'kpc': (
        np.array([0.5, 1.0, -1.0, 0.0, 3.0, -0.499, -1.0]),
        np.array([3.0, 10.0, 2.0, 3.0, 10.0, 1.0, 3.0]),
    ),
    # priorB / 8p_ZhaoPlumCOM_v3, i.e. that set's config.0.json with the
    # conditioning dimension (stellar_log_rstar, last column) dropped.
    'rstar': (
        np.array([0.5, 1.0, -1.0, 0.0, 3.0, -0.499, -1.0]),
        np.array([3.0, 10.0, 2.0, 3.0, 10.0, 1.0, 3.0]),
    ),
}
PRIOR_MIN, PRIOR_MAX = _PRIOR_BOUNDS[RADIUS_UNITS]

# Columns the box holds in r_star units but the model and tsnpe/sims.py
# want in kpc. Empty under 'rstar' - that emptiness is what makes
# to_kpc()/to_rstar_units() the identity, with no branching below.
RSTAR_SCALED_PARAM_NAMES = ('dm_log_rdm',) if RADIUS_UNITS == 'kpc' else ()
RSTAR_SCALED_INDICES = tuple(
    PARAM_NAMES.index(name) for name in RSTAR_SCALED_PARAM_NAMES)


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


def sample_conditioning(
    target: TargetData, n_samples: int, n_sigma: float = 5.0,
) -> np.ndarray:
    """Draw the conditioning dimension from its *prior*.

    Uniform in log10(r_half), i.e. exactly the conditioning marginal
    prior_box hands sample_prior_box. The single definition of the
    conditioning prior, so the proposal samplers cannot drift into a
    different one: an importance weight of 1{...}/q is only correct when
    the conditioning draws come from this, and drawing from anything else
    silently tilts the proposal in the conditioning dimension.

    Not the spectroscopic likelihood, deliberately. That belongs to the
    posterior (see _gaussian_conditioning_draws); the proposal is drawn
    from the prior, and the measured r_half is the unreliable quantity
    this whole conditioning scheme exists to keep out of it.

    Args:
        target: Target snapshot resolving the conditioning bounds.
        n_samples: Number of draws.
        n_sigma: Passed to `conditioning_bounds`; must match the
            `prior_n_sigma` the proposal samplers use.

    Returns:
        (n_samples,) log10(r_half [kpc]) draws.
    """
    cond_min, cond_max = conditioning_bounds(target, n_sigma=n_sigma)
    return np.random.uniform(cond_min, cond_max, n_samples)


def in_prior_box(
    theta_kpc: np.ndarray, target: TargetData, n_sigma: float = 5.0,
) -> np.ndarray:
    """Mask of model-unit rows lying inside the prior box.

    Converted back to box space first, which is where the box is defined.
    Under RADIUS_UNITS='kpc' a cut on kpc theta cannot express it:
    `dm_log_rdm` is an offset from the conditioning value, so its kpc
    bounds slide with each row's own conditioning draw. Testing kpc values
    against fixed bounds admits rows whose offset is outside [0, 3] --
    i.e. r_dm below r_star, which the box exists to forbid. Under
    'rstar' the conversion is the identity and the cut is direct.

    Args:
        theta_kpc: (N, 8) rows in the model's radius units,
            ALL_PARAM_NAMES order.
        target: Target snapshot resolving the conditioning bounds.
        n_sigma: Passed to `conditioning_bounds`.

    Returns:
        (N,) boolean mask.
    """
    theta_box = to_rstar_units(np.asarray(theta_kpc, dtype=float))
    prior_min, prior_max = prior_box(target, n_sigma=n_sigma)
    lo = prior_min.numpy()
    hi = prior_max.numpy()
    return np.all((theta_box >= lo) & (theta_box <= hi), axis=-1)


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


def sample_prior_box(
    n_samples: int, target: TargetData, n_sigma: float = 5.0,
) -> np.ndarray:
    """Draw `n_samples` uniform samples from the prior box, in box space.

    Box space means radii in r_star units; pass the result through
    `to_kpc` before it reaches the model or the simulator.

    Args:
        n_samples: Number of samples to draw.
        target: Target snapshot used to resolve the conditioning bounds.
        n_sigma: Passed to `conditioning_bounds`.

    Returns:
        (n_samples, 8) ndarray of box-space samples.
    """
    prior_min, prior_max = prior_box(target, n_sigma=n_sigma)
    dist = torch.distributions.Uniform(prior_min, prior_max)
    return dist.sample((n_samples,)).numpy()


def to_kpc(theta_rstar: np.ndarray) -> np.ndarray:
    """Map box-space samples to the units the model and simulator use.

    `theta_rstar[:, i]` for `i in RSTAR_SCALED_INDICES` holds
    log10(r / r_star), an offset from the conditioning value; the kpc
    value is offset + conditioning. Under RADIUS_UNITS='rstar' there are
    no such columns and this is the identity (a copy).

    Args:
        theta_rstar: (N, 8) box-space samples.

    Returns:
        (N, 8) samples in the model's radius units.
    """
    theta_kpc = np.asarray(theta_rstar, dtype=float).copy()
    cond = theta_kpc[:, CONDITIONING_INDEX]
    for i in RSTAR_SCALED_INDICES:
        theta_kpc[:, i] = theta_kpc[:, i] + cond
    return theta_kpc


def to_rstar_units(theta_kpc: np.ndarray) -> np.ndarray:
    """Inverse of `to_kpc`: map model-unit samples back to box space.

    Args:
        theta_kpc: (N, 8) samples in the model's radius units.

    Returns:
        (N, 8) box-space samples.
    """
    theta_rstar = np.asarray(theta_kpc, dtype=float).copy()
    cond = theta_rstar[:, CONDITIONING_INDEX]
    for i in RSTAR_SCALED_INDICES:
        theta_rstar[:, i] = theta_rstar[:, i] - cond
    return theta_rstar


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
