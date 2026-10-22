"""10 parameters: priorA in a flattened DM halo viewed at inclination i.

Adds the halo axis ratio q = z/x and cos(i), the latter uniform on
[0, 1] (isotropic orientations; i in [0, pi/2] by the i -> pi - i
symmetry). The halo is a Multipole expansion of an axisymmetric
Spheroid; a triaxial halo is unreachable since agama's action finder is
axisymmetric-only. See study_flattened_halo_qi.ipynb. Units are priorA's.
"""

import agama
import numpy as np

from . import common
from ..spec import ModelSpec

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_beta0', 'df_log_ra', 'stellar_log_rstar', 'dm_q', 'cos_inc',
)
# Reason: Multipole's default lmax=8 carries 0.15 dex of density error at
# q=0.4 over the radii the tracer stars occupy, which would show up as a
# q-dependent bias in the rho_s label. lmax=24 costs nothing measurable.
LMAX = 24


def rotate(posvel: np.ndarray, inclination: float) -> np.ndarray:
    """Intrinsic frame xyz -> observer frame XYZ (Z along the line of
    sight) by the Euler angle beta = inclination, radians."""
    if inclination == 0.0:
        return posvel
    rot = agama.makeRotationMatrix(0.0, inclination, 0.0)
    out = np.empty_like(posvel)
    out[:, :3] = posvel[:, :3].dot(rot.T)
    out[:, 3:] = posvel[:, 3:].dot(rot.T)
    return out


def simulate(num_stars, params):
    """Sample one galaxy in the observer frame."""
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar,
     q, cos_inc) = params
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar * r_dm
    r_a = 10 ** log_ra * r_star
    rho_s = 10 ** log_rhos
    try:
        dm_potential = agama.Potential(
            type='Multipole',
            density=agama.Density(
                type='Spheroid', alpha=alpha, beta=beta, gamma=gamma,
                scaleRadius=r_dm, densityNorm=rho_s, axisRatioZ=q,
                outercutoffRadius=max(50, 10 * r_dm)),
            lmax=LMAX, mmax=0, symmetry='axisymmetric')
        stellar_density = agama.Density(
            type='Plummer', mass=1, scaleRadius=r_star)
        dist_function = agama.DistributionFunction(
            type='QuasiSpherical', potential=dm_potential,
            density=stellar_density, beta0=beta0, r_a=r_a)
        galaxy_model = agama.GalaxyModel(dm_potential, dist_function)
        posvel, _ = galaxy_model.sample(num_stars)
    except Exception:
        return None
    return rotate(posvel, np.arccos(cos_inc)), {}


def preprocess(posvel, params):
    """priorA cuts with priorA's r_star convention."""
    r_star = 10 ** params[7] * 10 ** params[3]
    return common.preprocess(posvel, r_star)


def columns(params, extras):
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar,
     q, cos_inc) = params
    log_rstar_kpc = log_rstar + log_rdm
    return {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rho0': log_rhos,
        'df_beta0': beta0, 'df_log_ra': log_ra,
        'stellar_log_ra_kpc': log_ra + log_rstar_kpc,
        'stellar_log_rstar': log_rstar,
        'stellar_log_rstar_kpc': log_rstar_kpc,
        'dm_q': q, 'cos_inc': cos_inc,
    }


SPEC = ModelSpec(
    name='10p_FlatZhaoPlumCOM',
    description='priorA + halo axis ratio q and viewing inclination',
    param_names=PARAM_NAMES,
    prior_min=(0.5, 1.0, -1.0, -2.0, 3.0, -0.499, -1.0, -3.0, 0.4, 0.0),
    prior_max=(3.0, 10.0, 2.0, 2.0, 10.0, 1.0, 3.0, 0.0, 1.0, 1.0),
    column_names=tuple(columns(np.zeros(10), {})),
    labels=PARAM_NAMES[:7] + ('dm_q', 'cos_inc'),
    cond_labels=('stellar_log_rstar_kpc',),
    datasets=('10p_FlatZhaoPlumCOM',),
    simulate=simulate, preprocess=preprocess, columns=columns,
)
