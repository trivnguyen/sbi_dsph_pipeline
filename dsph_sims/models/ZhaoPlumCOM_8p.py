"""8-parameter Zhao (alpha-beta-gamma) halo + Plummer tracer, prior A.

Units: r_dm in kpc, r_star in units of r_dm, r_a in units of r_star. The
halo's outer cutoff is floored at 50 kpc and there is no velocity-
dispersion cut. Sampling order: (dm_alpha, dm_beta, dm_gamma, dm_log_rdm,
dm_log_rho0, df_beta0, df_log_ra, stellar_log_rstar).

Made 8p_ZhaoPlumCOM and 8p_ZhaoPlumCOM_v2; reproduces
legacy/simulate_8params_process_priorA.py bit for bit (test_specs.py).
The NPE conditions on stellar_log_rstar_kpc, a stored column derived
from the sampled r_star, so the model predicts dm_log_rdm in kpc while
tsnpe's box holds it relative to r_star - see `TSNPE`.
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
PRIOR_MIN = (0.5, 1.0, -1.0, -2.0, 3.0, -0.499, -1.0, -3.0)
PRIOR_MAX = (3.0, 10.0, 2.0, 2.0, 10.0, 1.0, 3.0, 0.0)


def simulate(num_stars, params):
    """Sample 6D stellar kinematics of one galaxy.

    Returns:
        ((num_stars, 6) pos/vel, {}) or None if agama rejects the model.
    """
    alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar = params
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar * r_dm
    r_a = 10 ** log_ra * r_star
    rho_s = 10 ** log_rhos
    try:
        dm_potential = agama.Potential(
            type='Spheroid', alpha=alpha, beta=beta, gamma=gamma,
            scaleRadius=r_dm, densityNorm=rho_s,
            outercutoffRadius=max(50, 10 * r_dm))
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
    r_star = 10 ** params[7] * 10 ** params[3]
    return common.preprocess(posvel, r_star)


def columns(params, extras):
    """Stored columns; r_star is sampled relative to r_dm, so its kpc
    value is derived."""
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra,
     log_rstar) = params
    log_rstar_kpc = log_rstar + log_rdm
    return {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rho0': log_rhos,
        'df_beta0': beta0, 'df_log_ra': log_ra,
        'stellar_log_ra_kpc': log_ra + log_rstar_kpc,
        'stellar_log_rstar': log_rstar,
        'stellar_log_rstar_kpc': log_rstar_kpc,
    }


def model_to_sim(theta_model):
    """Model units -> sampling space: the model predicts log10(r_dm/kpc)
    and is conditioned on log10(r_star/kpc); the simulator wants
    log10(r_star/r_dm) = log10(r_star/kpc) - log10(r_dm/kpc)."""
    theta = np.asarray(theta_model, dtype=float).copy()
    theta[:, 7] = theta[:, 7] - theta[:, 3]
    return theta


# tsnpe's box holds dm_log_rdm as log10(r_dm/r_star) = -stellar_log_rstar,
# so its bounds are the negated, swapped bounds of the sampled r_star axis.
TSNPE = TsnpePrior(
    param_names=LABELS,
    prior_min={**dict(zip(LABELS, PRIOR_MIN[:7])),
               'dm_log_rdm': 0.0 - PRIOR_MAX[7]},
    prior_max={**dict(zip(LABELS, PRIOR_MAX[:7])),
               'dm_log_rdm': 0.0 - PRIOR_MIN[7]},
    kpc_offset_names=('dm_log_rdm',),
    cond_columns=('stellar_log_rstar_kpc',),
    model_to_sim=model_to_sim,
)

SPEC = ModelSpec(
    name='8p_ZhaoPlumCOM',
    description='r_dm in kpc, r_star in r_dm units; 50 kpc cutoff floor; '
                'no vdisp cut',
    param_names=PARAM_NAMES, prior_min=PRIOR_MIN, prior_max=PRIOR_MAX,
    column_names=tuple(columns(np.zeros(8), {})),
    labels=LABELS, cond_labels=('stellar_log_rstar_kpc',),
    datasets=('8p_ZhaoPlumCOM', '8p_ZhaoPlumCOM_v2'),
    simulate=simulate, preprocess=preprocess, columns=columns,
    tsnpe=TSNPE,
)
