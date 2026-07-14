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
"""

import json
import logging
import sys
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

# In the packaged bundle, `tsnpe`/`jgnn`/`dsph_analysis` sit next to
# this file; in the repo, the tsnpe package lives in ../tsnpe.
_APP_DIR = Path(__file__).resolve().parent
for _p in (_APP_DIR, _APP_DIR.parent / 'tsnpe'):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

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
        posterior: (n_samples, n_params) physical-unit samples.
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
        labels=prior.ALL_PARAM_NAMES, color=o.get('color', '#2a78d6'),
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
