"""11 parameters: priorA with a full alpha-beta-gamma (Zhao) tracer.

Plummer is the (2, 5, 0) member, so this is a strict superset of priorA's
tracer. Bounds from study_abg_light_profile.ipynb: gamma_star >= 2 beta0
is a hard joint constraint (enforced at draw time and again in
`simulate`), beta_star >= 3.2 keeps the tracer mass finite, beta0 <= 0.75
and log_ra >= 0 follow from the constraint. Because of the constraint
the prior is NOT a box, which is one reason `tsnpe` is None here.
Units are priorA's.
"""

import agama
import numpy as np

from . import common
from ..spec import ModelSpec

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_beta0', 'df_log_ra', 'stellar_log_rstar', 'stellar_alpha',
    'stellar_beta', 'stellar_gamma',
)
I_BETA0, I_GAMMA_STAR = 5, 10


def constraint(params) -> bool:
    """Slope-anisotropy theorem, central form."""
    return params[I_GAMMA_STAR] >= 2 * params[I_BETA0]


def simulate(num_stars, params):
    """Sample one galaxy; extras carry its projected half-light radius."""
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar,
     alpha_star, beta_star, gamma_star) = params
    # Reason: belt-and-suspenders alongside the rejection at draw time -
    # the Eddington inversion silently returns a truncated DF here, so
    # the stars would not follow the label.
    if gamma_star < 2 * beta0:
        return None
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar * r_dm
    r_a = 10 ** log_ra * r_star
    rho_s = 10 ** log_rhos
    try:
        dm_potential = agama.Potential(
            type='Spheroid', alpha=alpha, beta=beta, gamma=gamma,
            scaleRadius=r_dm, densityNorm=rho_s,
            outercutoffRadius=max(50, 10 * r_dm))
        # Reason: mass=1 rather than densityNorm keeps the tracer
        # normalized to unit mass, so the shape parameters do not couple
        # to the normalization.
        stellar_density = agama.Density(
            type='Spheroid', alpha=alpha_star, beta=beta_star,
            gamma=gamma_star, scaleRadius=r_star, mass=1)
        r_half = common.half_light_radius(
            stellar_density, r_max=max(50.0, 100 * r_star))
        dist_function = agama.DistributionFunction(
            type='QuasiSpherical', potential=dm_potential,
            density=stellar_density, beta0=beta0, r_a=r_a)
        galaxy_model = agama.GalaxyModel(dm_potential, dist_function)
        posvel, _ = galaxy_model.sample(num_stars)
    except Exception:
        return None
    return posvel, {'r_half': r_half}


def preprocess(posvel, params):
    """priorA cuts with priorA's r_star convention."""
    r_star = 10 ** params[7] * 10 ** params[3]
    return common.preprocess(posvel, r_star)


def columns(params, extras):
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar,
     alpha_star, beta_star, gamma_star) = params
    log_rstar_kpc = log_rstar + log_rdm
    return {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rho0': log_rhos,
        'df_beta0': beta0, 'df_log_ra': log_ra,
        'df_log_ra_kpc': log_ra + log_rstar_kpc,
        'stellar_log_rstar': log_rstar,
        'stellar_log_rstar_kpc': log_rstar_kpc,
        'stellar_alpha': alpha_star, 'stellar_beta': beta_star,
        'stellar_gamma': gamma_star,
        'stellar_log_rhalf_kpc': np.log10(extras['r_half']),
    }


SPEC = ModelSpec(
    name='11p_ZhaoABGCOM',
    description='priorA with an alpha-beta-gamma tracer; gamma_* >= 2 beta0',
    param_names=PARAM_NAMES,
    prior_min=(0.5, 1.0, -1.0, -2.0, 3.0, -0.499, 0.0, -3.0, 0.5, 3.2, 0.0),
    prior_max=(3.0, 10.0, 2.0, 2.0, 10.0, 0.75, 3.0, 0.0, 3.0, 8.0, 1.5),
    column_names=tuple(columns(np.zeros(11), {'r_half': 1.0})),
    labels=PARAM_NAMES[:7],
    cond_labels=('stellar_log_rstar_kpc', 'stellar_alpha', 'stellar_beta',
                 'stellar_gamma'),
    datasets=('11p_ZhaoABGCOM',),
    simulate=simulate, preprocess=preprocess, columns=columns,
    constraint=constraint, constraint_text='stellar_gamma >= 2 * df_beta0',
)
