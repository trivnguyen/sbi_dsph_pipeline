"""9 parameters: priorA with a free tracer inner slope gamma_star.

Plummer outer shape held fixed (alpha_star = 2, beta_star = 5); Plummer
itself is gamma_star = 0. Two bounds differ from priorA, both from
study_abg_light_profile.ipynb: beta0 <= 0 (the slope-anisotropy theorem
gamma_star >= 2 beta0 must hold for every gamma_star in the box, or
agama's Eddington inversion silently clips the DF) and log_ra >= 0 (the
theorem's local form at intermediate radii). Units are priorA's.

R_h decouples from r_star once gamma_star is free (R_h/r_star spans
0.8-6 here), so the projected half-light radius is computed per galaxy
and stored as the conditioning column.
"""

import agama
import numpy as np

from . import common
from ..spec import ModelSpec

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_beta0', 'df_log_ra', 'stellar_log_rstar', 'stellar_gamma',
)
# Plummer outer shape, held fixed; only the inner slope is freed.
STELLAR_ALPHA, STELLAR_BETA = 2.0, 5.0


def simulate(num_stars, params):
    """Sample one galaxy; extras carry its projected half-light radius."""
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar,
     gamma_star) = params
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
        # normalized to unit mass, so gamma_star does not couple to the
        # normalization.
        stellar_density = agama.Density(
            type='Spheroid', alpha=STELLAR_ALPHA, beta=STELLAR_BETA,
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
     gamma_star) = params
    log_rstar_kpc = log_rstar + log_rdm
    return {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rho0': log_rhos,
        'df_beta0': beta0, 'df_log_ra': log_ra,
        'stellar_log_ra_kpc': log_ra + log_rstar_kpc,
        'stellar_log_rstar': log_rstar,
        'stellar_log_rstar_kpc': log_rstar_kpc,
        'stellar_gamma': gamma_star,
        'stellar_log_rhalf_kpc': np.log10(extras['r_half']),
    }


SPEC = ModelSpec(
    name='9p_ZhaoPlumGammaCOM',
    description='priorA + free tracer inner slope gamma_star (beta0 <= 0)',
    param_names=PARAM_NAMES,
    prior_min=(0.5, 1.0, -1.0, -2.0, 3.0, -0.499, 0.0, -3.0, 0.0),
    prior_max=(3.0, 10.0, 2.0, 2.0, 10.0, 0.0, 3.0, 0.0, 1.0),
    column_names=tuple(columns(np.zeros(9), {'r_half': 1.0})),
    labels=PARAM_NAMES[:7],
    cond_labels=('stellar_log_rhalf_kpc', 'stellar_gamma'),
    datasets=('9p_ZhaoPlumGammaCOM',),
    simulate=simulate, preprocess=preprocess, columns=columns,
)
