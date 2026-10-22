"""14 parameters: DF-free Jeans galaxies, independent beta(r) and beta'(r),
with beta(r) constrained to be non-decreasing (v2).

v2 differs from ZhaoPlumJeansBVH_14p_v1 only by the joint prior
constraint betainf >= beta0 (`constraint`): the anisotropy runs from
isotropic / tangential inside to radial outside, as in Osipkov-Merritt
and Cuddeford models and in simulated dwarfs in equilibrium. It is an
equilibrium prior, not a theorem: tidally stripped dwarfs develop
tangential outskirts, which v2 excludes. beta'(r) is left free (no
ordering result exists for the fourth-order anisotropy). With the BvH
form beta(r) is monotonic, so the constraint is exactly the sign of
betainf - beta0; in the symmetrised beta~ it is the same inequality.

Wraps sbi_twins/scripts/simulate_14params_jeans_bvh.py (the sbi_twins
project; see its README and notes/fourth_order_jeans_beta_prime.md). It is
the one family here that does **not** sample a distribution function
with agama, so most of what the other specs take for granted is different:

* Positions come from the Plummer tracer analytically, with the radial
  cut of the acceptance criteria built into the draw; velocities are the
  line-of-sight component only, drawn per star from the Jeans moments at
  its projected radius -- Gaussian at sigma_los(R) (`SPEC_TWIN`), or the
  Sanders & Evans kernel at (sigma, kappa)(R) with the fourth-order
  moments (`SPEC_KTWIN`). The vx, vy columns are zeros: a shard carries no
  proper-motion information and must not be trained with random
  projections (pre_transforms projection axis 2 only).
* Anisotropy is Baes & van Hese (2007), beta(r) = beta0 + (betainf -
  beta0) (r/r0)^n / (1 + (r/r0)^n), with flat priors on the symmetrised
  beta~ = beta / (2 - beta) as in GravSphere; the fourth-order beta'(r)
  gets its own four parameters (GravSphere2 Eq. 43) instead of the
  closure beta' = beta. Only the kurtosis draw sees beta'; in
  `SPEC_TWIN` its four labels are prior noise.
* Randomness is NumPy's, one generator per galaxy (`seeded=True`): the
  runner draws a seed after the star count exactly as the script did, so
  a dataset is reproducible from its config seed regardless of pool
  scheduling. Star counts are floored at 10 (`min_stars`).
* Rejection has reasons, recorded in config.<i>.json: 'nonfinite' (no
  Jeans solution), 'negative' (<v_los^4> < 0 on the grid inside 50 r_star
  or at a star -- no distribution function does that, and GravSphere2's
  likelihood rejects it too) and 'cuts'. The kurtosis draw also stores
  per-galaxy diagnostics (kurt_min, kurt_max, frac_clamped,
  theta4_negative) as columns next to the labels.
* Units are prior B/C's (r_star in kpc; r_dm, r0, r0' in r_star units);
  halo bounds are 8p_ZhaoPlumCOM_v4's. The stored columns add the plain
  beta0/betainf and the kpc radii so the profiles can be rebuilt without
  the conventions.

The helpers the script took from the sbi_twins package (Plummer
positions, twin velocities, kurtosis flags, the production cuts) are
inlined here, and the Sanders & Evans kernel is a verbatim copy in
models/sanders_evans.py, so nothing imports sbi_twins. The one runtime
dependency is the current `dsph_analysis.sph_model.BaesVanHeseJeans`,
imported lazily inside `simulate` so the rest of the registry (and
tsnpe, which imports it) does not need it.

Sampling order: (dm_alpha, dm_beta, dm_gamma, dm_log_rdm,
dm_log_rho0, df_betasym0, df_betasyminf, df_log_r0, df_n_aniso,
df_betasym0_prime, df_betasyminf_prime, df_log_r0_prime, df_n_aniso_prime,
stellar_log_rstar). No tsnpe adapter: 13 labels and a DF-free forward
model are outside what the truncation machinery has been checked on.
"""

import numpy as np

from . import sanders_evans
from ..spec import ModelSpec

PARAM_NAMES = (
    'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm', 'dm_log_rho0',
    'df_betasym0', 'df_betasyminf', 'df_log_r0', 'df_n_aniso',
    'df_betasym0_prime', 'df_betasyminf_prime', 'df_log_r0_prime',
    'df_n_aniso_prime', 'stellar_log_rstar',
)
LABELS = PARAM_NAMES[:13]

# The beta~ floor of -0.9 (beta = -18) is numerical: below it r^(2 beta0)
# leaves the float range on the radial grid.
PRIOR_MIN = {
    'dm_alpha': 0.5,
    'dm_beta': 2.5,
    'dm_gamma': -0.5,
    'dm_log_rdm': 0.0,
    'dm_log_rho0': 5.0,
    'df_betasym0': -0.9,
    'df_betasyminf': -0.9,
    'df_log_r0': -1.0,
    'df_n_aniso': 1.0,
    'df_betasym0_prime': -0.9,
    'df_betasyminf_prime': -0.9,
    'df_log_r0_prime': -1.0,
    'df_n_aniso_prime': 1.0,
    'stellar_log_rstar': -2.0,
}
PRIOR_MAX = {
    'dm_alpha': 3.0,
    'dm_beta': 10.0,
    'dm_gamma': 2.0,
    'dm_log_rdm': 2.5,
    'dm_log_rho0': 10.0,
    'df_betasym0': 1.0,
    'df_betasyminf': 1.0,
    'df_log_r0': 1.0,
    'df_n_aniso': 3.0,
    'df_betasym0_prime': 1.0,
    'df_betasyminf_prime': 1.0,
    'df_log_r0_prime': 1.0,
    'df_n_aniso_prime': 3.0,
    'stellar_log_rstar': 0.5,
}
PRIOR_MIN = tuple(PRIOR_MIN[name] for name in PARAM_NAMES)
PRIOR_MAX = tuple(PRIOR_MAX[name] for name in PARAM_NAMES)


I_BETASYM0, I_BETASYMINF = 5, 6
I_N_ANISO, I_N_ANISO_PRIME, I_LOG_RSTAR = 8, 12, 13
MIN_STARS = 10
# Reason: GravSphere2 requires non-negative fourth moments out to
# rcn = rmax / 2 = 50 R_half; R_half = r_star for a Plummer tracer.
NONNEG_MAX_RSTAR = 50.0
LOG_RHO_UNIT = 7.0  # sph_model works in 1e7 Msun / kpc^3
# sigma_r^2 grid of the Jeans model, in units of r_star (sph_model's
# GRID_MIN_RSTAR, GRID_MAX_RSTAR, N_GRID).
GRID_MIN_RSTAR, GRID_MAX_RSTAR, N_GRID = 1e-3, 1e3, 300
# The production acceptance cuts (sbi_twins.simulate).
MAX_RADIUS_RSTAR, MAX_RADIUS_KPC, MAX_SPEED = 10.0, 100.0, 200.0
MIN_KEPT_FRACTION = 0.5
KURT_FALLBACK = 3.0  # Gaussian where the fourth-order solution is absent
KTWIN_FLAGS = ('kurt_min', 'kurt_max', 'frac_clamped', 'theta4_negative')


def constraint(params) -> bool:
    """beta(r) non-decreasing: betainf >= beta0 (beta~ is monotonic in
    beta, so the symmetrised parameters carry the same inequality)."""
    return params[I_BETASYMINF] >= params[I_BETASYM0]


def sym_to_beta(beta_sym):
    """Symmetrised anisotropy beta~ = beta / (2 - beta) back to beta."""
    return 2.0 * beta_sym / (1.0 + beta_sym)


def to_jeans_theta(params):
    """Map one draw onto the GeneralizedOMJeans parameter vectors.

    Returns:
        (theta, theta_prime): theta = [log_rho_s, log_r_s, alp, bet, gam,
        log_r0, two_to_beta0, two_to_betainf, r_star, 0, 0, 0] and
        theta_prime = [log_r0', two_to_beta0', two_to_betainf'], radii
        in kpc and rho_s in units of 1e7 Msun/kpc^3.
    """
    (alpha, beta, gamma, log_rdm, log_rhos, betasym0, betasyminf, log_r0,
     _, betasym0_prime, betasyminf_prime, log_r0_prime, _,
     log_rstar) = params
    theta = np.array([
        log_rhos - LOG_RHO_UNIT, log_rdm + log_rstar, alpha, beta, gamma,
        log_r0 + log_rstar, 2.0 ** sym_to_beta(betasym0),
        2.0 ** sym_to_beta(betasyminf), 10 ** log_rstar, 0.0, 0.0, 0.0,
    ])
    theta_prime = np.array([
        log_r0_prime + log_rstar, 2.0 ** sym_to_beta(betasym0_prime),
        2.0 ** sym_to_beta(betasyminf_prime),
    ])
    return theta, theta_prime


def build_bvh_jeans(theta, n_aniso, theta_prime, n_aniso_prime):
    """The Baes & van Hese model with its grid scaled to r_star."""
    # Reason: lazy so the registry imports without the current
    # dsph_analysis (BaesVanHeseJeans is recent) on the path.
    from dsph_analysis.sph_model import BaesVanHeseJeans
    r_star = float(np.asarray(theta)[8])
    return BaesVanHeseJeans(
        theta, n_aniso=n_aniso, n_aniso_prime=n_aniso_prime,
        theta_prime=theta_prime, min_rgrid=GRID_MIN_RSTAR * r_star,
        max_rgrid=GRID_MAX_RSTAR * r_star, n_grid=N_GRID)


def max_radius(r_star):
    """The radial acceptance cut, kpc."""
    return min(MAX_RADIUS_RSTAR * r_star, MAX_RADIUS_KPC)


def plummer_positions(r_star, num_stars, rng):
    """3D positions from a Plummer profile inside the radial cut."""
    # Reason: the Plummer enclosed-mass fraction is m = (1 + a^2/r^2)^-1.5,
    # so inverting it gives exact radii; draw the fraction only up to the
    # value at the radial cut instead of rejecting.
    m_max = (1 + (r_star / max_radius(r_star)) ** 2) ** -1.5
    m = rng.uniform(0, m_max, num_stars)
    r = r_star / np.sqrt(m ** (-2.0 / 3.0) - 1.0)
    cos_theta = rng.uniform(-1, 1, num_stars)
    phi = rng.uniform(0, 2 * np.pi, num_stars)
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.column_stack([
        r * sin_theta * np.cos(phi), r * sin_theta * np.sin(phi),
        r * cos_theta])


def usable_kurtosis(kurt):
    """Clamp to the Sanders & Evans table; non-finite -> Gaussian."""
    kurt = np.where(np.isfinite(kurt), kurt, KURT_FALLBACK)
    return np.clip(kurt, sanders_evans.KURT_MIN, sanders_evans.KURT_MAX)


def twin_velocities(sigma, rng, kurt=None):
    """(0, 0, vz) per star: Gaussian at sigma, or Sanders & Evans at
    (sigma, kurt). The line of sight is z; vx, vy are placeholders."""
    vel = np.zeros((len(sigma), 3))
    if kurt is None:
        vel[:, 2] = rng.normal(0.0, sigma)
    else:
        vel[:, 2] = sanders_evans.sample(sigma, usable_kurtosis(kurt), rng)
    return vel


def kurtosis_flags(kurt):
    """kurt_min/kurt_max over the stars (kurt_min < 1 marks an unphysical
    fourth-order solution) and the fraction clamped by the table."""
    finite = kurt[np.isfinite(kurt)]
    outside = ((kurt < sanders_evans.KURT_MIN)
               | (kurt > sanders_evans.KURT_MAX) | ~np.isfinite(kurt))
    return dict(
        kurt_min=float(finite.min()) if finite.size else np.nan,
        kurt_max=float(finite.max()) if finite.size else np.nan,
        frac_clamped=float(outside.mean()))


def _simulate(num_stars, params, rng, kurtosis):
    """Positions from the Plummer tracer, LOS velocities from the Jeans
    moments; see the module docstring for the rejection reasons."""
    theta, theta_prime = to_jeans_theta(params)
    r_star = 10 ** params[I_LOG_RSTAR]
    model = build_bvh_jeans(
        theta, params[I_N_ANISO], theta_prime=theta_prime,
        n_aniso_prime=params[I_N_ANISO_PRIME])
    pos = plummer_positions(r_star, num_stars, rng)
    R = np.hypot(pos[:, 0], pos[:, 1])
    sigma2 = model.sigma2_los(R)
    sigma = np.sqrt(sigma2)
    if not np.all(np.isfinite(sigma)):
        return 'nonfinite'
    if not kurtosis:
        return np.hstack([pos, twin_velocities(sigma, rng)]), {}

    # Reason: the (beta' - beta) coupling can drive the fourth moments
    # negative, which no distribution function does; GravSphere2 returns
    # -inf for such models, so the twin must not produce them either.
    r_check = model.r_grid[model.r_grid <= NONNEG_MAX_RSTAR * r_star]
    sigma4 = model.sigma4_los(np.concatenate([R, r_check]))
    if not np.all(np.isfinite(sigma4)):
        return 'nonfinite'
    if np.any(sigma4 < 0):
        return 'negative'
    kurt = sigma4[:len(R)] / sigma2 ** 2
    flags = kurtosis_flags(kurt)
    flags['theta4_negative'] = float(
        np.any(model.sigma4_theta(r_check) < 0))
    return np.hstack([pos, twin_velocities(sigma, rng, kurt)]), flags


def simulate_twin(num_stars, params, rng):
    """Gaussian velocities at sigma_los(R); beta' unused."""
    return _simulate(num_stars, params, rng, kurtosis=False)


def simulate_ktwin(num_stars, params, rng):
    """Sanders & Evans velocities at (sigma, kappa)(R) with beta'."""
    return _simulate(num_stars, params, rng, kurtosis=True)


def preprocess(posvel, params):
    """The production cuts: r < min(10 r_star, 100 kpc) and |v| < 200
    km/s; the galaxy is rejected if fewer than half its stars survive."""
    r_star = 10 ** params[I_LOG_RSTAR]
    rad3d = np.linalg.norm(posvel[:, :3], axis=1)
    vel3d = np.linalg.norm(posvel[:, 3:6], axis=1)
    mask = (rad3d < max_radius(r_star)) & (vel3d < MAX_SPEED)
    if mask.sum() < int(MIN_KEPT_FRACTION * len(posvel)):
        return None
    return posvel[mask]


def _columns(params, extras, flag_names):
    (alpha, beta, gamma, log_rdm, log_rhos, betasym0, betasyminf, log_r0,
     n_aniso, betasym0_prime, betasyminf_prime, log_r0_prime,
     n_aniso_prime, log_rstar) = params
    cols = {
        'dm_alpha': alpha, 'dm_beta': beta, 'dm_gamma': gamma,
        'dm_log_rdm': log_rdm, 'dm_log_rdm_kpc': log_rdm + log_rstar,
        'dm_log_rho0': log_rhos,
        'df_betasym0': betasym0, 'df_betasyminf': betasyminf,
        'df_beta0': sym_to_beta(betasym0),
        'df_betainf': sym_to_beta(betasyminf),
        'df_log_r0': log_r0, 'df_log_r0_kpc': log_r0 + log_rstar,
        'df_n_aniso': n_aniso,
        'df_betasym0_prime': betasym0_prime,
        'df_betasyminf_prime': betasyminf_prime,
        'df_beta0_prime': sym_to_beta(betasym0_prime),
        'df_betainf_prime': sym_to_beta(betasyminf_prime),
        'df_log_r0_prime': log_r0_prime,
        'df_log_r0_prime_kpc': log_r0_prime + log_rstar,
        'df_n_aniso_prime': n_aniso_prime,
        'stellar_log_rstar': log_rstar, 'stellar_log_rstar_kpc': log_rstar,
    }
    cols.update({k: extras[k] for k in flag_names})
    return cols


def columns_twin(params, extras):
    """Labels, conditioning and the derived beta / kpc columns."""
    return _columns(params, extras, ())


def columns_ktwin(params, extras):
    """`columns_twin` plus the kurtosis-draw diagnostics."""
    return _columns(params, extras, KTWIN_FLAGS)


_COMMON = dict(
    param_names=PARAM_NAMES, prior_min=PRIOR_MIN, prior_max=PRIOR_MAX,
    labels=LABELS, cond_labels=('stellar_log_rstar_kpc',),
    preprocess=preprocess, min_stars=MIN_STARS, seeded=True,
    constraint=constraint, constraint_text='df_betasyminf >= df_betasym0',
)

SPEC_TWIN = ModelSpec(
    name='14p_ZhaoPlumJeansBVH_v2_twin',
    description='DF-free Jeans twin: Plummer positions, Gaussian LOS '
                "velocities at sigma_los(R); non-decreasing BvH beta(r)",
    column_names=tuple(columns_twin(np.zeros(14), {})),
    datasets=(), simulate=simulate_twin, columns=columns_twin, **_COMMON,
)

SPEC_KTWIN = ModelSpec(
    name='14p_ZhaoPlumJeansBVH_v2_ktwin',
    description='DF-free Jeans kurtosis twin: Sanders & Evans LOS velocities '
                "at (sigma, kappa)(R); non-decreasing beta(r), free beta'(r)",
    column_names=tuple(columns_ktwin(
        np.zeros(14), dict.fromkeys(KTWIN_FLAGS, 0.0))),
    datasets=(), simulate=simulate_ktwin, columns=columns_ktwin, **_COMMON,
)
