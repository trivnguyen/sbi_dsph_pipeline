"""8-parameter Zhao (alpha-beta-gamma) halo + Plummer tracer, prior B.

Prior A with the radius convention changed: r_dm in units of r_star,
r_star in kpc, r_a in units of r_star. The halo's outer cutoff is 10 r_dm
(no floor) and there is no velocity-dispersion cut. Sampling order:
(dm_alpha, dm_beta, dm_gamma, dm_log_rdm, dm_log_rho0, df_beta0,
df_log_ra, stellar_log_rstar).

Made 8p_ZhaoPlumCOM_v3; reproduces legacy/simulate_8params_process_priorB.py
bit for bit (test_specs.py). Model units are the sampling space, so
tsnpe's box maps are the identity - see `TSNPE`.
"""

import agama
import numpy as np

from . import common
from ..spec import ModelSpec, TsnpePrior

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_beta0', 'df_log_ra', 'stellar_log_rstar',
)
LABELS = PARAM_NAMES[:7]
PRIOR_MIN = (0.5, 1.0, -1.0, 0.0, 3.0, -0.499, -1.0, -2.0)
PRIOR_MAX = (3.0, 10.0, 2.0, 3.0, 10.0, 1.0, 3.0, 0.0)


def simulate(num_stars, params):
    """Sample 6D stellar kinematics of one galaxy.

    Returns:
        ((num_stars, 6) pos/vel, {}) or None if agama rejects the model.
    """
    alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar = params
    r_star = 10 ** log_rstar
    r_dm = 10 ** log_rdm * r_star
    r_a = 10 ** log_ra * r_star
    rho_s = 10 ** log_rhos
    try:
        dm_potential = agama.Potential(
            type='Spheroid', alpha=alpha, beta=beta, gamma=gamma,
            scaleRadius=r_dm, densityNorm=rho_s,
            outerCutoffRadius=10 * r_dm,
            cutoffStrength=2.)
        stellar_density = agama.Density(
            type='Plummer', mass=1, scaleRadius=r_star)
        dist_function = agama.DistributionFunction(
            type='QuasiSpherical', potential=dm_potential,
            density=stellar_density, beta0=beta0, r_a=r_a)
        galaxy_model = agama.GalaxyModel(dm_potential, dist_function)
        posvel, _ = galaxy_model.sample(num_stars)
    except Exception:
        return None
    return posvel, {}


def preprocess(posvel, params):
    """Per-star speed/radius cuts only; no dispersion cut."""
    r_star = 10 ** params[7]
    return common.preprocess(posvel, r_star)


def columns(params, extras):
    """Stored columns; r_dm is sampled relative to r_star, so its kpc
    value is derived."""
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra,
     log_rstar) = params
    return {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rdm_kpc': log_rdm + log_rstar,
        'dm_log_rho0': log_rhos, 'df_beta0': beta0, 'df_log_ra': log_ra,
        'stellar_log_ra_kpc': log_ra + log_rstar,
        'stellar_log_rstar': log_rstar,
        'stellar_log_rstar_kpc': log_rstar,
    }


def model_to_sim(theta_model):
    """Model units are the sampling space: dm_log_rdm is log10(r_dm/r_star)
    on both sides and the conditioning column is log10(r_star/kpc)."""
    return np.asarray(theta_model, dtype=float).copy()


TSNPE = TsnpePrior(
    param_names=LABELS,
    prior_min=dict(zip(LABELS, PRIOR_MIN[:7])),
    prior_max=dict(zip(LABELS, PRIOR_MAX[:7])),
    kpc_offset_names=(),
    cond_columns=('stellar_log_rstar', 'stellar_log_rstar_kpc'),
    model_to_sim=model_to_sim,
)

SPEC = ModelSpec(
    name='8p_ZhaoPlumCOM_v3',
    description='r_dm in r_star units, r_star in kpc; cutoff 10 '
                'r_dm; no vdisp cut',
    param_names=PARAM_NAMES, prior_min=PRIOR_MIN, prior_max=PRIOR_MAX,
    column_names=tuple(columns(np.zeros(8), {})),
    labels=LABELS, cond_labels=('stellar_log_rstar_kpc',),
    datasets=('8p_ZhaoPlumCOM_v3',),
    simulate=simulate, preprocess=preprocess, columns=columns,
    tsnpe=TSNPE,
)
