"""Standalone inference/plot-data layer for the web app.

Self-contained counterpart of plotting/posterior_diagnostics.py: model
loading, Jeans-profile computation, Wolf mass, the corner-plot PNG, and
the JSON payload the frontend's interactive (Plotly) profile panels
consume. Kept free of repo-local machinery (register_run, wandb,
absolute style/catalog paths) so `package.py` can ship it in a portable
bundle - see webapp/README.md.

The model directory format is what npe/train_npe.py leaves behind: a
Lightning `.ckpt` (norm_dict embedded in its hyperparameters) plus a
`config_snapshot.json` next to it or up to a few parent levels above.

Radius units
------------
`tsnpe.proposal.sample_posterior` returns draws in the units the
*model* predicts, which for the radius columns is not kpc:
`df_log_ra` is log10(r_a / r_star) under every model, and
`dm_log_rdm` is too under `radius_units='rstar'` (priorB). The Jeans
code wants kpc, so `to_physical_kpc` sits between the two and
everything downstream of it (`_jeans_worker`, `_theta_from_params`)
is kpc-only. Which convention a given checkpoint uses is not recorded
in it - see `build_prior` - so it is declared per model, once, and
app.py carries it alongside the flow it belongs to.
"""

import json
import logging
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

warnings.filterwarnings('ignore', category=UserWarning)
# vdisp/vkurtosis run with verbose=False, but emcee's too-short-chain
# autocorrelation warning is a logging call that verbose doesn't cover.
logging.getLogger('emcee').setLevel(logging.ERROR)

import astropy.constants as aconst
import astropy.units as auni
import corner
import matplotlib.pyplot as plt
import numpy as np
import torch
from ml_collections import ConfigDict

# `tsnpe`/`jgnn`/`dsph_analysis` come from the bundle's vendored
# copies or from the repo checkout - see app_paths for which, and why
# the two are mutually exclusive rather than one falling back to the
# other.
import app_paths

app_paths.setup()

from tsnpe import prior
from tsnpe.model_io import build_npe
from tsnpe.proposal import sample_posterior  # re-exported for app.py

from dsph_analysis import kinematic_io
from dsph_analysis.sph_model import GeneralizedOMJeans

R_VEC_KPC = np.logspace(-2, 1, 50)

PROFILE_KEYS = ('rho', 'mass', 'beta', 'sigma', 'kappa')

# Inner (central) density log-slope gamma; the UI can restrict the
# posterior to a sub-range of it (see restrict_gamma).
GAMMA_PARAM = 'dm_gamma'

# Column indices into a posterior row, in `prior.ALL_PARAM_NAMES` order:
# (dm_alpha, dm_beta, dm_gamma, dm_log_rdm, dm_log_rho0, df_beta0,
#  df_log_ra, stellar_log_rstar).
I_LOG_RDM = prior.PARAM_NAMES.index('dm_log_rdm')
I_LOG_RA = prior.PARAM_NAMES.index('df_log_ra')
I_COND = prior.CONDITIONING_INDEX

# The radius columns, once to_physical_kpc has run. Used to label the
# corner plot and the CSV, and to decide which panels get the r_star
# reference marker.
RADIUS_PARAM_INDICES = (I_LOG_RDM, I_LOG_RA)

# Posterior column names once converted to kpc. The `_kpc` suffixes
# mark the columns that conversion moves, since a downloaded CSV
# outlives any note about which convention it holds.
KPC_PARAM_NAMES = [
    f'{name}_kpc'
    if i in RADIUS_PARAM_INDICES or i == I_COND else name
    for i, name in enumerate(prior.ALL_PARAM_NAMES)
]

CORNER_LABELS = [
    f'{name} [kpc]' if i in RADIUS_PARAM_INDICES or i == I_COND
    else name
    for i, name in enumerate(prior.ALL_PARAM_NAMES)
]

# Stands in for r_a=inf: GeneralizedOMJeans.dbeta_dr divides by r_a**2,
# so a literal inf produces 0 * inf = nan in kurtosis_los.
_LARGE_FINITE_R_A_KPC = 1e6

# G in kpc*(km/s)^2/Msun, as in Wolf et al. 2010's mass estimator.
_G_KPC_KMS2_MSUN = aconst.G.to(
    auni.kpc * (auni.km / auni.s) ** 2 / auni.Msun).value


def _find_config_snapshot(
    checkpoint_path: Path, max_levels: int = 3,
) -> Optional[Path]:
    """Look for config_snapshot.json next to the checkpoint or a few
    parents up (train_npe.py writes it at the run's workdir root, with
    the checkpoints/ directory below it).
    """
    current = checkpoint_path.resolve().parent
    for _ in range(max_levels):
        candidate = current / 'config_snapshot.json'
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def load_model(model_dir: str, checkpoint_filename: str, device):
    """Load a pretrained NPE checkpoint fully offline.

    Args:
        model_dir: Directory holding the checkpoint file, with
            config_snapshot.json alongside it (or a few parents up).
        checkpoint_filename: Checkpoint file name within model_dir.
        device: torch device to move the model to.

    Returns:
        (model, norm_dict, pre_transforms_config).

    Raises:
        FileNotFoundError: If the checkpoint or its config snapshot is
            missing.
    """
    checkpoint_path = Path(model_dir) / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'No checkpoint at {checkpoint_path}')
    snapshot_path = _find_config_snapshot(checkpoint_path)
    if snapshot_path is None:
        raise FileNotFoundError(
            f'No config_snapshot.json found near {checkpoint_path} - '
            'the model directory must hold the .ckpt plus the training '
            'config snapshot written by npe/train_npe.py.')

    full_config = ConfigDict(json.loads(snapshot_path.read_text()))
    checkpoint = torch.load(
        checkpoint_path, map_location='cpu', weights_only=False)
    norm_dict = checkpoint['hyper_parameters']['norm_dict']

    model = build_npe(full_config.model, pre_transforms=None,
                      norm_dict=None)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    return model, norm_dict, full_config.pre_transforms


def to_physical_kpc(
    posterior: np.ndarray, radius_units: str,
) -> np.ndarray:
    """Put every radius column of a posterior into log10(r / kpc).

    `sample_posterior` returns draws in the *model's own* radius units
    (it only un-normalizes with norm_dict; it never converts), so this
    has to run before any row reaches GeneralizedOMJeans. Two separate
    conversions, both keyed off the conditioning column
    `stellar_log_rstar`, which is always log10(r_star / kpc):

    * `dm_log_rdm` is log10(r_dm / r_star) under `radius_units='rstar'`
      (priorB, e.g. 8p_ZhaoPlumCOM_v3) and already log10(r_dm / kpc)
      under `'kpc'` (priorA, e.g. 8p_ZhaoPlumCOM), so only the former
      is shifted.
    * `df_log_ra` is log10(r_a / r_star) under **both** conventions -
      the training simulators for priorA and priorB both wrote an
      explicit `r_a = 10 ** log_ra * r_star` (see tsnpe/tsnpe/sims.py,
      and npe/simulate_8params_process_prior{A,B}.py, which is what
      actually built the two datasets). It is therefore *always*
      shifted, whatever the model.

    Kept deliberately identical to `plotting/posterior_diagnostics.py`
    and `npe_infer.sampling.to_physical_kpc` in the repo - all three
    read the same checkpoints, so they must agree here or their
    profiles diverge.

    Args:
        posterior: (N, 8) draws in the model's own radius units,
            columns in `prior.ALL_PARAM_NAMES` order.
        radius_units: The model's convention, 'kpc' or 'rstar'.

    Returns:
        (N, 8) draws with every radius column in log10(r / kpc).

    Raises:
        ValueError: If `radius_units` is not recognized.
    """
    if radius_units not in prior.RADIUS_UNITS_CHOICES:
        raise ValueError(
            f'radius_units={radius_units!r} not recognized; must be '
            f'one of {prior.RADIUS_UNITS_CHOICES}')
    physical = np.asarray(posterior, dtype=float).copy()
    log_rstar = physical[:, I_COND]
    if radius_units == 'rstar':
        physical[:, I_LOG_RDM] = physical[:, I_LOG_RDM] + log_rstar
    physical[:, I_LOG_RA] = physical[:, I_LOG_RA] + log_rstar
    return physical


def build_prior(
    radius_units: str, prior_min: Optional[dict] = None,
    prior_max: Optional[dict] = None,
) -> 'prior.Prior':
    """Construct the prior box `sample_posterior` cuts against.

    A checkpoint cannot tell you its own radius convention - the
    training config snapshot npe/train_npe.py writes has no `prior`
    block - so `radius_units` is a deliberate per-model declaration,
    exactly as in npe_inference's `configs/models.py`. Getting it wrong
    does not raise; it silently shifts `dm_log_rdm` by a factor of
    r_star, which is why app.py refuses to guess.

    Args:
        radius_units: 'kpc' (priorA) or 'rstar' (priorB).
        prior_min: Box-space lower bounds by parameter name. None takes
            `tsnpe.prior`'s priorA defaults, which the whole
            8-parameter family shares.
        prior_max: Upper bounds, same rules.

    Returns:
        The `tsnpe.prior.Prior` for this model.

    Raises:
        ValueError: If `radius_units` or either bounds dict is invalid.
    """
    return prior.Prior(prior_min=prior_min, prior_max=prior_max,
                       radius_units=radius_units)


def restrict_gamma(
    posterior: np.ndarray, gamma_min: Optional[float],
    gamma_max: Optional[float],
) -> np.ndarray:
    """Keep only posterior draws whose inner slope gamma is within
    [gamma_min, gamma_max] (either bound may be None to leave that side
    open).

    This is a straight rejection cut on the already-drawn samples, so
    the returned set is smaller than requested; the caller reports the
    surviving count. Passing both bounds as None returns the input
    unchanged.
    """
    if gamma_min is None and gamma_max is None:
        return posterior
    idx = list(prior.ALL_PARAM_NAMES).index(GAMMA_PARAM)
    gamma = posterior[:, idx]
    mask = np.ones(len(posterior), dtype=bool)
    if gamma_min is not None:
        mask &= gamma >= gamma_min
    if gamma_max is not None:
        mask &= gamma <= gamma_max
    return posterior[mask]


def _theta_from_params(alp, bet, gam, r_s, r_a, beta0, rho_s, rh):
    """Build a GeneralizedOMJeans theta vector from physical-unit
    params. GeneralizedOMJeans expects log_rho_s in units of 1e7
    Msun/kpc^3 (hence the -7); rho/M consumers undo it with * 1e7.
    """
    r_a = _LARGE_FINITE_R_A_KPC if np.isinf(r_a) else r_a
    return np.array([
        np.log10(rho_s) - 7, np.log10(r_s), alp, bet, gam,
        np.log10(r_a), 2.0 ** beta0, 2.0, rh, 0, 0, 0,
    ])


def _jeans_worker(args):
    # Top-level so ProcessPoolExecutor can pickle it.
    i, row, r_vec = args
    alp, bet, gam, log_rdm, log_rhos, beta0, log_ra, log_rstar = row
    theta = _theta_from_params(
        alp=alp, bet=bet, gam=gam, r_s=10 ** log_rdm, r_a=10 ** log_ra,
        beta0=beta0, rho_s=10 ** log_rhos, rh=10 ** log_rstar)
    m = GeneralizedOMJeans(theta)
    return (i, m.rho(r_vec) * 1e7, m.M(r_vec) * 1e7, m.beta(r_vec),
            np.sqrt(m.sigma2_los(r_vec)), m.kurtosis_los(r_vec))


def calc_jeans_profiles(
    posterior: np.ndarray, r_vec: np.ndarray, n_samples: int,
    n_workers: int,
) -> dict:
    """Compute Jeans profiles from a subsample of posterior draws.

    Returns:
        dict of (n_samples, len(r_vec)) sample arrays: `rho` [Msun/
        kpc^3], `mass` [Msun], `beta`, `sigma` [km/s], `kappa`.
    """
    idx = np.random.choice(
        len(posterior), size=min(n_samples, len(posterior)),
        replace=False)
    n = len(idx)
    out = {name: np.zeros((n, len(r_vec)))
           for name in ('rho', 'mass', 'beta', 'sigma', 'kappa')}

    work = [(i, posterior[j], r_vec) for i, j in enumerate(idx)]
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_jeans_worker, w) for w in work]
        for fut in as_completed(futures):
            i, rho, mass, beta, sigma, kappa = fut.result()
            for name, values in zip(
                    ('rho', 'mass', 'beta', 'sigma', 'kappa'),
                    (rho, mass, beta, sigma, kappa)):
                out[name][i] = values
    return out


def calc_wolf_mass(
    vlos_kms: np.ndarray, vlos_err_kms: np.ndarray, rhalf_kpc: float,
) -> dict:
    """Classical dynamical mass at the Wolf radius (Wolf et al. 2010,
    eq. 2): M_1/2 = 4/G * sigma_los^2 * r_1/2, with sigma_los the
    global error-deconvolved LOS dispersion. Model-independent, so it's
    a sanity check against the posterior's own mass profile.
    """
    from dsph_analysis import vdisp

    r_wolf_kpc = (4.0 / 3.0) * rhalf_kpc
    samples = vdisp.fit_vdisp_los(vlos_kms, vlos_err_kms, verbose=False)
    sigma_los_kms = np.exp(samples[:, 1])
    mass = 4.0 / _G_KPC_KMS2_MSUN * sigma_los_kms ** 2 * r_wolf_kpc
    return dict(r_wolf_kpc=r_wolf_kpc, mass_wolf_samples=mass)


def load_literature_mass_wolf(key: str) -> Optional[tuple]:
    """Published Wolf mass for `key` from the bundled
    local_volume_database snapshot, or None if it has no entry.

    Returns:
        (log10_mass_msun, minus_err, plus_err) or None.
    """
    try:
        meta = kinematic_io.load_meta(key)
    except ValueError:
        return None
    if meta.log_mass_wolf is None or np.isnan(meta.log_mass_wolf):
        return None
    return (float(meta.log_mass_wolf), float(meta.log_mass_wolf_em),
            float(meta.log_mass_wolf_ep))


def _corner_ranges(posterior: np.ndarray) -> list:
    """Per-parameter (min, max) for corner, widened for any column with
    no spread.

    A parameter can collapse to a single value - most often
    stellar_log_rstar when the target's rhalf is given with zero
    uncertainty, so the conditioning distribution is a delta. corner
    then raises "column(s) have no dynamic range"; giving it an
    explicit padded range for those columns keeps the plot working.
    """
    ranges = []
    for col in posterior.T:
        lo, hi = float(np.min(col)), float(np.max(col))
        if hi - lo <= 1e-9 * (abs(hi) + 1e-9):
            pad = abs(hi) * 0.05 + 1e-3
            lo, hi = lo - pad, hi + pad
        ranges.append((lo, hi))
    return ranges


def plot_corner(posterior: np.ndarray, save_path: Path, title: str,
                options: Optional[dict] = None) -> None:
    """Save a corner plot of physical-unit posterior samples.

    Args:
        posterior: (n_samples, n_params) samples with every radius
            column already in log10(r / kpc), i.e. to_physical_kpc
            output - the r_star marker below reads the conditioning
            column directly, so a model-unit array would mislabel it.
        save_path: PNG output path.
        title: Figure suptitle prefix.
        options: Optional corner.corner styling overrides. Recognized
            keys (all optional; omitting `options` reproduces the
            library-default look):
                smooth: gaussian smoothing width in bins for the 2D
                    histograms and the 1D marginals (None/<=0 = off).
                bins: number of bins per parameter (default 20).
                contours: 'default' (corner's own contour lines),
                    'filled', 'lines', or 'off'.
                plot_datapoints: draw the individual samples (bool).
                color: line/point color (hex).
    """
    o = options or {}
    smooth = o.get('smooth')
    smooth = smooth if (smooth and smooth > 0) else None
    contours = o.get('contours', 'default')

    kwargs = dict(
        labels=CORNER_LABELS, color=o.get('color', '#2a78d6'),
        show_titles=True, title_fmt='.2f', quantiles=[0.16, 0.5, 0.84],
        range=_corner_ranges(posterior), bins=int(o.get('bins') or 20),
        smooth=smooth, smooth1d=smooth,
        plot_datapoints=bool(o.get('plot_datapoints', True)),
    )
    # 'default' passes no contour kwargs, so corner uses its own look.
    if contours == 'off':
        kwargs.update(plot_contours=False, no_fill_contours=True,
                      fill_contours=False)
    elif contours == 'lines':
        kwargs.update(plot_contours=True, fill_contours=False)
    elif contours == 'filled':
        kwargs.update(plot_contours=True, fill_contours=True)

    fig = corner.corner(posterior, **kwargs)

    # Both radius parameters are bounded relative to r_star rather than
    # in absolute kpc, so where a draw sits relative to it is the part
    # that carries information. Mark it on exactly those panels.
    log_rstar = float(np.median(posterior[:, I_COND]))
    ndim = posterior.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    marker = dict(color='#d1495b', linestyle='--', linewidth=1.2,
                  zorder=5)
    for i in RADIUS_PARAM_INDICES:
        for row in range(i, ndim):       # panels with this param on x
            axes[row, i].axvline(log_rstar, **marker)
        for col in range(i):             # panels with this param on y
            axes[i, col].axhline(log_rstar, **marker)
    fig.legend(
        handles=[plt.Line2D([], [], **marker)],
        labels=[r'median $\log_{10}(r_\star/\mathrm{kpc})$'
                f' = {log_rstar:.2f}'],
        loc='upper right', frameon=False, fontsize=12)

    fig.suptitle(f'{title} (N={len(posterior)})', y=1.02, fontsize=13)
    fig.savefig(save_path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def _round_sig(a: np.ndarray, sig: int = 4) -> np.ndarray:
    """Round to `sig` significant figures - shrinks the JSON payload of
    the raw sample arrays (4 sig figs is far finer than any percentile
    band needs) without changing scale across the many decades these
    profiles span.
    """
    a = np.asarray(a, dtype=float)
    out = np.array(a)
    nz = np.isfinite(a) & (a != 0)
    mag = np.floor(np.log10(np.abs(a[nz])))
    factor = 10.0 ** (sig - 1 - mag)
    out[nz] = np.round(a[nz] * factor) / factor
    return out


def _binned_points(profile: dict, value_key: str) -> dict:
    """Reshape a vdisp/vkurtosis binned-fit result for JSON."""
    return dict(
        R=np.asarray(profile['R_mid']).tolist(),
        R_em=np.asarray(profile['R_em']).tolist(),
        R_ep=np.asarray(profile['R_ep']).tolist(),
        val=np.asarray(profile[value_key]).tolist(),
        em=np.asarray(profile[f'{value_key}_em']).tolist(),
        ep=np.asarray(profile[f'{value_key}_ep']).tolist(),
    )


def profiles_payload(
    r_vec: np.ndarray, jeans: dict, vdisp_profile: dict,
    vkurtosis_profile: dict, wolf: dict,
) -> dict:
    """Assemble the interactive-profile JSON the frontend plots.

    Args:
        r_vec: Radius grid [kpc].
        jeans: calc_jeans_profiles result.
        vdisp_profile: vdisp.calc_vdisp_los_binned result.
        vkurtosis_profile: vkurtosis.calc_kurtosis_los_binned result.
        wolf: calc_wolf_mass result, plus a `literature` entry (see
            load_literature_mass_wolf).

    Returns:
        JSON-friendly dict: `r_kpc`; `samples`, the raw per-draw profile
        arrays (shape n_draws x len(r_vec)) for each panel (`rho`,
        `mass`, `beta`, `sigma`, `kappa`) so the frontend can recompute
        median + credible bands at any percentile level without a
        re-run; the binned data points for the sigma/kappa panels; and
        the Wolf markers.
    """
    mass_lo, mass_med, mass_hi = np.percentile(
        wolf['mass_wolf_samples'], [16, 50, 84])
    literature = wolf.get('literature')
    return dict(
        r_kpc=r_vec.tolist(),
        samples={name: _round_sig(jeans[name]).tolist()
                 for name in PROFILE_KEYS},
        binned=dict(
            sigma=_binned_points(vdisp_profile, 'sigma'),
            kappa=_binned_points(vkurtosis_profile, 'kappa'),
        ),
        wolf=dict(
            r_wolf_kpc=float(wolf['r_wolf_kpc']),
            mass_median=float(mass_med),
            mass_em=float(mass_med - mass_lo),
            mass_ep=float(mass_hi - mass_med),
            literature=(None if literature is None
                        else list(literature)),
        ),
    )
