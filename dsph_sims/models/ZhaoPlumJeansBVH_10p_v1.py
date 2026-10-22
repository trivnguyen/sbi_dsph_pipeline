"""10 parameters: DF-free Jeans galaxies with Baes & van Hese beta(r).

The reduced version of ZhaoPlumJeansBVH_14p: the fourth-order anisotropy
keeps the closure beta'(r) = beta(r), so the four beta' parameters are
gone and no <v_los^4> < 0 rejection is needed. Wraps
sbi_twins/scripts/simulate_10params_jeans_bvh.py, which it reproduces bit
for bit (tests/test_specs.py). Everything else is as in the 14p spec and
imported from it (Plummer positions, LOS-only velocities with vx = vy = 0,
NumPy randomness with one seed per galaxy, the production cuts, kurtosis
diagnostics as columns). Prior C's anisotropy is the n = 2, betainf = 1
slice of this family.

Units are prior B/C's (r_star in kpc; r_dm and r0 in r_star units); halo
bounds are 8p_ZhaoPlumCOM_v4's. Sampling order: (dm_alpha, dm_beta,
dm_gamma, dm_log_rdm, dm_log_rho0, df_betasym0, df_betasyminf, df_log_r0,
df_n_aniso, stellar_log_rstar). `SPEC_KTWIN` is the production spec;
`SPEC_TWIN` (Gaussian velocities) is registered for testing. No tsnpe
adapter.
"""

import numpy as np

from ..spec import ModelSpec
from .ZhaoPlumJeansBVH_14p_v1 import (KTWIN_FLAGS, LOG_RHO_UNIT, MIN_STARS,
                                   build_bvh_jeans, kurtosis_flags,
                                   plummer_positions, sym_to_beta,
                                   twin_velocities)
from .ZhaoPlumJeansBVH_14p_v1 import preprocess as _preprocess_14p

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_betasym0', 'df_betasyminf', 'df_log_r0', 'df_n_aniso',
    'stellar_log_rstar',
)
LABELS = PARAM_NAMES[:9]
PRIOR_MIN = (0.5, 2.5, -0.5, 0.0, 5.0, -0.9, -0.9, -1.0, 1.0, -2.0)
PRIOR_MAX = (3.0, 10.0, 2.0, 2.5, 10.0, 1.0, 1.0, 1.0, 3.0, 0.5)
I_N_ANISO, I_LOG_RSTAR = 8, 9


def to_jeans_theta(params):
    """The 12-element GeneralizedOMJeans vector, radii in kpc and rho_s in
    units of 1e7 Msun/kpc^3."""
    (alpha, beta, gamma, log_rdm, log_rhos, betasym0, betasyminf, log_r0,
     _, log_rstar) = params
    return np.array([
        log_rhos - LOG_RHO_UNIT, log_rdm + log_rstar, alpha, beta, gamma,
        log_r0 + log_rstar, 2.0 ** sym_to_beta(betasym0),
        2.0 ** sym_to_beta(betasyminf), 10 ** log_rstar, 0.0, 0.0, 0.0,
    ])


def _simulate(num_stars, params, rng, kurtosis):
    """Plummer positions, LOS velocities from the Jeans moments with
    beta' = beta; None if the Jeans solution is not finite."""
    model = build_bvh_jeans(to_jeans_theta(params), params[I_N_ANISO],
                            theta_prime=None, n_aniso_prime=None)
    pos = plummer_positions(10 ** params[I_LOG_RSTAR], num_stars, rng)
    R = np.hypot(pos[:, 0], pos[:, 1])
    sigma2 = model.sigma2_los(R)
    sigma = np.sqrt(sigma2)
    kurt = model.sigma4_los(R) / sigma2 ** 2 if kurtosis else None
    if not np.all(np.isfinite(sigma)):
        return None
    posvel = np.hstack([pos, twin_velocities(sigma, rng, kurt)])
    return posvel, kurtosis_flags(kurt) if kurtosis else {}


def simulate_twin(num_stars, params, rng):
    """Gaussian velocities at sigma_los(R)."""
    return _simulate(num_stars, params, rng, kurtosis=False)


def simulate_ktwin(num_stars, params, rng):
    """Sanders & Evans velocities at (sigma, kappa)(R)."""
    return _simulate(num_stars, params, rng, kurtosis=True)


def preprocess(posvel, params):
    """The production cuts (see the 14p spec); r_star is params[9] here."""
    return _preprocess_14p(posvel, np.append(params[:9], [0, 0, 0, 0,
                                                         params[9]]))


def _columns(params, extras, flag_names):
    (alpha, beta, gamma, log_rdm, log_rhos, betasym0, betasyminf, log_r0,
     n_aniso, log_rstar) = params
    cols = {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rdm_kpc': log_rdm + log_rstar,
        'dm_log_rho0': log_rhos,
        'df_betasym0': betasym0, 'df_betasyminf': betasyminf,
        'df_beta0': sym_to_beta(betasym0),
        'df_betainf': sym_to_beta(betasyminf),
        'df_log_r0': log_r0, 'df_log_r0_kpc': log_r0 + log_rstar,
        'df_n_aniso': n_aniso,
        'stellar_log_rstar': log_rstar, 'stellar_log_rstar_kpc': log_rstar,
    }
    cols.update({k: extras[k] for k in flag_names})
    return cols


def columns_twin(params, extras):
    """Labels, conditioning and the derived beta / kpc columns."""
    return _columns(params, extras, ())


def columns_ktwin(params, extras):
    """`columns_twin` plus the kurtosis-draw diagnostics."""
    return _columns(params, extras, KTWIN_FLAGS[:3])


_COMMON = dict(
    param_names=PARAM_NAMES, prior_min=PRIOR_MIN, prior_max=PRIOR_MAX,
    labels=LABELS, cond_labels=('stellar_log_rstar_kpc',),
    preprocess=preprocess, min_stars=MIN_STARS, seeded=True,
)

SPEC_TWIN = ModelSpec(
    name='10p_ZhaoPlumJeansBVH_twin',
    description="DF-free Jeans twin, BvH beta(r) with beta' = beta; "
                'Gaussian LOS velocities at sigma_los(R)',
    column_names=tuple(columns_twin(np.zeros(10), {})),
    datasets=(), simulate=simulate_twin, columns=columns_twin, **_COMMON,
)

SPEC_KTWIN = ModelSpec(
    name='10p_ZhaoPlumJeansBVH_ktwin',
    description="DF-free Jeans kurtosis twin, BvH beta(r) with beta' = "
                'beta; Sanders & Evans LOS velocities at (sigma, kappa)(R)',
    column_names=tuple(columns_ktwin(
        np.zeros(10), dict.fromkeys(KTWIN_FLAGS[:3], 0.0))),
    datasets=(), simulate=simulate_ktwin, columns=columns_ktwin, **_COMMON,
)
