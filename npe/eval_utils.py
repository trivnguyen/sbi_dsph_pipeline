"""
Shared machinery for evaluating a trained NPE run on a held-out test set.

The three `eval_*.ipynb` notebooks are thin wrappers around this module:
each one loads a run, samples its posterior over the test shard, and calls
the plotting helpers below.

Two conventions matter and are easy to get wrong:

* Posterior samples and `theta` both live in the *normalized* space the
  flow was trained in. Everything returned by `sample_posterior` is
  un-normalized back to physical units first, using the `norm_dict` stored
  in the checkpoint - never a freshly computed one.
* The test shard is the one shard training never read. `train_npe.py`
  reads shards `init .. init + num_datasets - 1`, and the simulators were
  run long enough to leave a spare, so shard `init + num_datasets` is
  genuinely held out rather than a re-split of training data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tarp
import torch
from ml_collections import ConfigDict

from jgnn import datasets
from jgnn.models import NPE, GNNEmbedding
from jgnn.transforms import build_transformation

# Root holding every dwarf-galaxy NPE run: <WORKDIR>/<wandb_project>/
# <run_id>/{checkpoints, config_snapshot.json}. Moved under the project
# tree (was /scratch/$USER/trained_models/dsph_npe, and
# /scratch/$USER/trained_models/npe before that); the older streams-era
# projects (4p/6p/9p_AAU*) were left behind at those paths, so point
# NPE_MODEL_WORKDIR at one to eval them. Same env var and default as
# npe_inference's npe_infer.paths.MODEL_WORKDIR, so the two pipelines
# cannot drift.
WORKDIR = os.environ.get(
    'NPE_MODEL_WORKDIR',
    '/scratch/tvnguyen/projects/sbi_dsph/trained_models/npe')
ARTIFACT_CACHE = '/scratch/tvnguyen/cache/wandb_artifacts'

# Reason: default.mplstyle sets text.usetex, so a raw feature name like
# dm_log_rho0 fails at draw time on the underscore. Every name that can
# reach an axis label needs an entry here.
LABEL_TEX = {
    'dm_alpha': r'$\alpha_{\rm DM}$',
    'dm_beta': r'$\beta_{\rm DM}$',
    'dm_gamma': r'$\gamma_{\rm DM}$',
    'dm_log_rdm': r'$\log r_{\rm dm}$',
    'dm_log_rho0': r'$\log \rho_s$',
    'df_beta0': r'$\beta_0$',
    'df_log_ra': r'$\log r_a$',
    'dm_q': r'$q$',
    'cos_inc': r'$\cos i$',
    'stellar_alpha': r'$\alpha_\star$',
    'stellar_beta': r'$\beta_\star$',
    'stellar_gamma': r'$\gamma_\star$',
    'stellar_log_rstar': r'$\log a_\star / r_{\rm dm}$',
    'stellar_log_rstar_kpc': r'$\log a_\star$',
    'stellar_log_rhalf_kpc': r'$\log R_h$',
    'stellar_log_ra_kpc': r'$\log r_a$ [kpc]',
}


def tex(name: str) -> str:
    """
    Render a feature name as a LaTeX-safe axis label.

    Args:
        name: Feature name as stored in the HDF5 file.

    Returns:
        A LaTeX string safe to pass to matplotlib under usetex.
    """
    return LABEL_TEX.get(name, name.replace('_', r'\_'))


def use_style() -> None:
    """Load the user's matplotlib style file."""
    plt.style.use(os.path.expanduser('~/default.mplstyle'))


def find_run_dir(project: str, run_id: str | None = None) -> Path:
    """
    Locate a training run's output directory.

    Args:
        project: `config.wandb_project`, the directory under WORKDIR.
        run_id: Specific wandb run id. If None, and exactly one run
            exists, that one is used; otherwise the most recent.

    Returns:
        Path to the run directory holding `checkpoints/` and
        `config_snapshot.json`.

    Raises:
        FileNotFoundError: If no run directory exists for `project`.
    """
    root = Path(WORKDIR) / project
    if run_id is not None:
        return root / run_id
    runs = [p for p in root.iterdir() if (p / 'checkpoints').is_dir()]
    if not runs:
        raise FileNotFoundError(f'No runs with checkpoints under {root}')
    runs.sort(key=lambda p: p.stat().st_mtime)
    if len(runs) > 1:
        print(f'{len(runs)} runs under {root}; using newest {runs[-1].name}')
    return runs[-1]


def _build_npe(config: ConfigDict, ckpt_path: str | Path):
    """
    Rebuild an NPE model and load a checkpoint's weights into it.

    `NPE.load_from_checkpoint` alone is not enough: the embedding network
    and pre-transforms are excluded from the saved hyperparameters, so a
    bare load silently yields an identity embedding. Both are rebuilt
    here from the config instead.

    Args:
        config: The run's full config, giving model architecture and
            pre-transform settings.
        ckpt_path: Path to the `.ckpt` file.

    Returns:
        Tuple of (model in eval mode on the best available device, the
        checkpoint's own norm_dict).

    Raises:
        RuntimeError: If the checkpoint's weights do not match the
            architecture rebuilt from `config`.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    norm_dict = ckpt['hyper_parameters']['norm_dict']
    print(f'loaded {Path(ckpt_path).name} (epoch {ckpt.get("epoch", "?")})')

    embedding_nn = GNNEmbedding(
        input_size=config.model.input_size,
        gnn_args=config.model.embedding.gnn,
        mlp_args=config.model.embedding.mlp,
        conditional_mlp_args=config.model.embedding.get(
            'conditional_mlp', None),
        optimizer_args=None, scheduler_args=None, pre_transforms=None,
    )
    pre_transforms = build_transformation(
        norm_dict=norm_dict, **config.pre_transforms)
    model = NPE(
        input_size=config.model.input_size,
        output_size=config.model.output_size,
        flows_args=config.model.flows,
        embedding_nn=embedding_nn,
        optimizer_args=None, scheduler_args=None,
        norm_dict=norm_dict, pre_transforms=pre_transforms,
    )
    missing, unexpected = model.load_state_dict(
        ckpt['state_dict'], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'state_dict mismatch: {len(missing)} missing, '
            f'{len(unexpected)} unexpected (first: '
            f'{(missing + unexpected)[:3]})')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return model.to(device).eval(), norm_dict


def load_npe(
    run_dir: str | Path, checkpoint: str = 'last.ckpt',
) -> tuple[NPE, ConfigDict, dict]:
    """
    Rebuild a trained NPE model from a local run directory.

    Args:
        run_dir: Directory containing `checkpoints/` and
            `config_snapshot.json`.
        checkpoint: File name inside `checkpoints/` to load.

    Returns:
        Tuple of (model in eval mode, the run's full config, the
        checkpoint's own norm_dict).
    """
    run_dir = Path(run_dir)
    config = ConfigDict(
        json.loads((run_dir / 'config_snapshot.json').read_text()))
    model, norm_dict = _build_npe(
        config, run_dir / 'checkpoints' / checkpoint)
    return model, config, norm_dict


def load_npe_wandb(
    run_path: str, version: str = 'best',
) -> tuple[NPE, ConfigDict, dict]:
    """
    Rebuild a trained NPE model from a wandb run.

    Needed for runs whose local checkpoint directory no longer exists.
    Requires wandb credentials (`~/.netrc`) and network access.

    Args:
        run_path: Wandb run path, `entity/project/run_id`.
        version: Artifact version to fetch, e.g. 'best' or 'latest'.

    Returns:
        Tuple of (model in eval mode, the run's full config, the
        checkpoint's own norm_dict).
    """
    from jgnn.utils import fetch_wandb_checkpoint

    # Reason: fetch_wandb_checkpoint calls artifact.download() with no
    # root, which lands in ./artifacts relative to the cwd - i.e. inside
    # the repo when run from a notebook here. Fetch from a scratch cache
    # instead, which also means repeat calls reuse the download.
    cache = Path(ARTIFACT_CACHE)
    cache.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        os.chdir(cache)
        ckpt_path, config = fetch_wandb_checkpoint(
            run_path=run_path, version=version)
        ckpt_path = Path(ckpt_path).resolve()
    finally:
        os.chdir(cwd)

    print(f'fetched {run_path} ({version})')
    model, norm_dict = _build_npe(config, ckpt_path)
    return model, config, norm_dict


def load_test_set(
    config: ConfigDict,
    norm_dict: dict,
    shard: int | None = None,
    max_graphs: int | None = None,
):
    """
    Build a dataloader over the shard training never read.

    Args:
        config: The run's config, giving data location and label names.
        norm_dict: The checkpoint's norm_dict, reused verbatim so the
            test data is normalized exactly as training data was.
        shard: Shard index to use. Defaults to the first one past the
            training range.
        max_graphs: Cap on the number of test galaxies.

    Returns:
        Tuple of (dataloader, test shard index).
    """
    init = config.get('init', 0)
    if shard is None:
        shard = init + config.num_datasets
    node_feats, graph_feats = datasets.read_datasets(
        config.data_root, config.data_name, num_datasets=1, init=shard,
        is_directory=True, concat=True)
    n_avail = len(graph_feats['num_stars'])
    print(f'test shard data.{shard}.h5: {n_avail} galaxies '
          f'(training read shards {init}-{init + config.num_datasets - 1})')

    loader, _ = datasets.cartesian.prepare_test_dataloader(
        node_feats, graph_feats, config.labels,
        cond_labels=config.get('cond_labels', None),
        batch_size=config.eval_batch_size,
        num_workers=0, norm_dict=norm_dict, max_graphs=max_graphs,
        pre_transform_kwargs=dict(config.pre_transforms),
    )
    return loader, shard


def sample_posterior(
    model: NPE, loader, num_samples: int = 1000, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draw posterior samples for every galaxy in a loader.

    The model's own stochastic pre-transforms (radial selection,
    velocity uncertainty) are applied, exactly as during training, so
    the calibration statements below refer to the same data-generating
    process the model was trained for.

    Args:
        model: Trained NPE model in eval mode.
        loader: Test dataloader from `load_test_set`.
        num_samples: Posterior draws per galaxy.
        seed: Seed for the pre-transform randomness.

    Returns:
        Tuple of (samples, truth) in physical units, with shapes
        (n_galaxies, num_samples, n_params) and (n_galaxies, n_params).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    truth = torch.cat([b.theta.reshape(b.num_graphs, -1)
                       for b in loader], dim=0)
    samples = model.sample_from_loader(loader, num_samples, verbose=True)

    loc = torch.tensor(model.norm_dict['theta_loc'], dtype=torch.float32)
    scale = torch.tensor(model.norm_dict['theta_scale'], dtype=torch.float32)
    samples = (samples.cpu() * scale + loc).numpy()
    truth = (truth.cpu() * scale + loc).numpy()
    return samples, truth


def extract_cond(loader, norm_dict: dict) -> np.ndarray:
    """
    Pull the conditioning values back out of a loader, un-normalized.

    Args:
        loader: Test dataloader from `load_test_set`.
        norm_dict: The checkpoint's norm_dict, whose `cond_loc`/
            `cond_scale` were used to normalize them.

    Returns:
        Array of shape (n_galaxies, n_cond) in physical units.
    """
    cond = torch.cat([b.cond.reshape(b.num_graphs, -1) for b in loader])
    loc = torch.tensor(norm_dict['cond_loc'], dtype=torch.float32)
    scale = torch.tensor(norm_dict['cond_scale'], dtype=torch.float32)
    return (cond.cpu() * scale + loc).numpy()


def priorb_rdm_to_kpc(
    samples: np.ndarray, truth: np.ndarray, cond: np.ndarray,
    names: list[str], cond_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Put a Prior B run's `dm_log_rdm` into Prior A's units (kpc).

    The two prior schemes disagree on exactly one label. Prior A stores
    `dm_log_rdm` as log10(r_dm / kpc); Prior B stores it as
    log10(r_dm / r_star). Since r_star in kpc is a conditioning input and
    therefore known exactly per galaxy, the conversion is an additive
    shift of every posterior draw by a constant:

        log10(r_dm / kpc) = log10(r_dm / r_star) + log10(r_star / kpc)

    Every other label (the two DM slopes, gamma, log rho_s, beta0, and
    log_ra which is in units of r_star under both schemes) already means
    the same thing in both.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        cond: Conditioning values from `extract_cond`.
        names: Label names, must contain 'dm_log_rdm'.
        cond_names: Conditioning names, must contain
            'stellar_log_rstar_kpc'.

    Returns:
        Tuple of (samples, truth) with `dm_log_rdm` converted to kpc.

    Raises:
        ValueError: If the required label or conditioning name is absent.
    """
    if 'dm_log_rdm' not in names:
        raise ValueError("no 'dm_log_rdm' label to convert")
    if 'stellar_log_rstar_kpc' not in cond_names:
        raise ValueError(
            "Prior B conversion needs 'stellar_log_rstar_kpc' in "
            f"cond_labels, got {cond_names}")
    i = names.index('dm_log_rdm')
    shift = cond[:, cond_names.index('stellar_log_rstar_kpc')]

    samples = samples.copy()
    truth = truth.copy()
    samples[:, :, i] += shift[:, None]
    truth[:, i] += shift
    return samples, truth


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_corner(
    samples: np.ndarray, truth: np.ndarray, names: list[str],
    index: int = 0, out_path: str | None = None,
):
    """
    Corner plot of one galaxy's posterior against its true parameters.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        names: Feature names, one per parameter.
        index: Which test galaxy to show.
        out_path: If given, save the figure here.

    Returns:
        The matplotlib figure.
    """
    import corner

    fig = corner.corner(
        samples[index], truths=truth[index],
        labels=[tex(n) for n in names],
        show_titles=True, title_fmt='.2f',
        quantiles=[0.16, 0.5, 0.84],
        truth_color='C3', color='C0',
        plot_datapoints=False, fill_contours=True,
        levels=(0.68, 0.95),
        figsize=(2.2 * len(names), 2.2 * len(names)),
    )
    fig.suptitle(f'test galaxy {index}', y=1.0)
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig


def plot_pred_vs_true(
    samples: np.ndarray, truth: np.ndarray, names: list[str],
    n_show: int = 400, out_path: str | None = None,
):
    """
    Posterior median against truth, with a 68% credible interval.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        names: Feature names, one per parameter.
        n_show: Number of galaxies to draw error bars for; the reported
            statistics always use every galaxy.
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, per-parameter statistics dict).
    """
    median = np.median(samples, axis=1)
    lo, hi = np.percentile(samples, [16, 84], axis=1)

    n_par = len(names)
    n_col = min(4, n_par)
    n_row = int(np.ceil(n_par / n_col))
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.6 * n_col, 4.4 * n_row))
    axes = np.atleast_1d(axes).ravel()

    stats = {}
    sel = slice(None) if n_show is None else slice(0, n_show)
    for i, name in enumerate(names):
        ax = axes[i]
        ax.errorbar(
            truth[sel, i], median[sel, i],
            yerr=[median[sel, i] - lo[sel, i], hi[sel, i] - median[sel, i]],
            fmt='o', ms=3, alpha=0.35, lw=1, color='C0')
        span = [truth[:, i].min(), truth[:, i].max()]
        ax.plot(span, span, 'k--', lw=2, alpha=0.6)

        resid = median[:, i] - truth[:, i]
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((truth[:, i] - truth[:, i].mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        width = np.mean(hi[:, i] - lo[:, i])
        prior_width = truth[:, i].max() - truth[:, i].min()
        stats[name] = dict(
            r2=float(r2), bias=float(resid.mean()),
            scatter=float(resid.std()), width68=float(width),
            width_over_prior=float(width / prior_width))
        ax.set_title(rf'{tex(name)}:  $R^2 = {r2:.2f}$')
        ax.set_xlabel('true')
        ax.set_ylabel('posterior median')

    for ax in axes[n_par:]:
        ax.set_visible(False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig, stats


def plot_tarp(
    samples: np.ndarray, truth: np.ndarray, out_path: str | None = None,
):
    """
    TARP expected-coverage test of the joint posterior.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, maximum absolute coverage deviation).
    """
    ecp_boot, alpha = tarp.get_tarp_coverage(
        samples.transpose(1, 0, 2).astype(np.float64),
        truth.astype(np.float64),
        norm=True, metric='euclidean', references='random', bootstrap=True)
    ecp, err = ecp_boot.mean(0), ecp_boot.std(0)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.6, label='ideal')
    ax.plot(alpha, ecp, color='C0', lw=2.5, label='measured')
    for k in (1, 2, 3):
        ax.fill_between(alpha, ecp - k * err, ecp + k * err,
                        color='C0', alpha=0.18, lw=0)
    ax.set_xlabel('credibility level')
    ax.set_ylabel('expected coverage')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig, float(np.max(np.abs(ecp - alpha)))


def plot_rank_marginals(
    samples: np.ndarray, truth: np.ndarray, names: list[str],
    n_bins: int = 20, out_path: str | None = None,
):
    """
    Per-parameter rank histograms (simulation-based calibration).

    The rank of the truth among the posterior draws is uniform for a
    calibrated marginal. A U shape means the posterior is too narrow, a
    dome means too wide, and a slope means it is biased.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        names: Feature names, one per parameter.
        n_bins: Number of histogram bins.
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, per-parameter reduced chi-square against
        uniform).
    """
    n_gal, n_draw, n_par = samples.shape
    ranks = (samples < truth[:, None, :]).sum(axis=1)

    n_col = min(4, n_par)
    n_row = int(np.ceil(n_par / n_col))
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(4.6 * n_col, 3.8 * n_row))
    axes = np.atleast_1d(axes).ravel()

    # Expected count per bin, and its 1-sigma binomial spread.
    expected = n_gal / n_bins
    sigma = np.sqrt(n_gal * (1 / n_bins) * (1 - 1 / n_bins))

    chi2 = {}
    edges = np.linspace(0, n_draw, n_bins + 1)
    for i, name in enumerate(names):
        ax = axes[i]
        counts, _ = np.histogram(ranks[:, i], bins=edges)
        ax.stairs(counts, edges, fill=True, alpha=0.7, color='C0')
        ax.axhline(expected, color='k', ls='--', lw=2, alpha=0.6)
        ax.axhspan(expected - 2 * sigma, expected + 2 * sigma,
                   color='k', alpha=0.12, lw=0)
        chi2[name] = float(
            np.sum((counts - expected) ** 2 / expected) / (n_bins - 1))
        ax.set_title(rf'{tex(name)}:  $\chi^2_\nu = {chi2[name]:.1f}$')
        ax.set_xlabel('rank of truth')
        ax.set_ylabel('count')
        ax.set_ylim(0, None)

    for ax in axes[n_par:]:
        ax.set_visible(False)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig, chi2


def plot_model_comparison(
    stats_a: dict, stats_b: dict, names: list[str],
    label_a: str, label_b: str, out_path: str | None = None,
):
    """
    Compare two models' credible-interval widths parameter by parameter.

    Args:
        stats_a: `plot_pred_vs_true` statistics for the first model.
        stats_b: The same for the second model.
        names: Feature names, one per parameter.
        label_a: Legend label for the first model.
        label_b: Legend label for the second model.
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, dict mapping name to the b/a width ratio).
    """
    wa = np.array([stats_a[n]['width_over_prior'] for n in names])
    wb = np.array([stats_b[n]['width_over_prior'] for n in names])
    ra = np.array([stats_a[n]['r2'] for n in names])
    rb = np.array([stats_b[n]['r2'] for n in names])
    ratio = dict(zip(names, wb / wa))

    idx = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(8.2 * 2, 6.0))

    axes[0].barh(idx - 0.2, wa, height=0.38, color='C0', label=label_a)
    axes[0].barh(idx + 0.2, wb, height=0.38, color='C3', label=label_b)
    axes[0].axvline(1.0, color='k', ls=':', lw=2)
    axes[0].set_xlabel('68\\% width / prior range')
    axes[0].set_xlim(0, 1.05)
    axes[0].legend(loc='lower right')

    axes[1].barh(idx - 0.2, ra, height=0.38, color='C0', label=label_a)
    axes[1].barh(idx + 0.2, rb, height=0.38, color='C3', label=label_b)
    axes[1].axvline(0.0, color='k', ls=':', lw=2)
    axes[1].set_xlabel(r'$R^2$')

    for ax in axes:
        ax.set_yticks(idx)
        ax.set_yticklabels([tex(n) for n in names])
        ax.invert_yaxis()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
    return fig, ratio


def _rank_chi2(proj_samples: np.ndarray, proj_truth: np.ndarray,
               n_bins: int) -> float:
    """
    Reduced chi-square of a 1-D rank histogram against uniform.

    Args:
        proj_samples: Projected posterior draws, shape (n_gal, n_draw).
        proj_truth: Projected truths, shape (n_gal,).
        n_bins: Number of histogram bins.

    Returns:
        Reduced chi-square; 1 is ideal.
    """
    n_gal, n_draw = proj_samples.shape
    ranks = (proj_samples < proj_truth[:, None]).sum(axis=1)
    counts, _ = np.histogram(ranks, bins=np.linspace(0, n_draw, n_bins + 1))
    expected = n_gal / n_bins
    return float(np.sum((counts - expected) ** 2 / expected) / (n_bins - 1))


def plot_correlation_matrices(
    samples: np.ndarray, truth: np.ndarray, names: list[str],
    out_path: str | None = None,
):
    """
    What degeneracies the posterior reports, and whether they are right.

    Two matrices. The left one is the posterior correlation averaged over
    test galaxies - purely descriptive, it says which parameters the model
    thinks are degenerate. The right one is the correlation of *whitened*
    residuals: for each galaxy the error `truth - posterior mean` is
    rescaled by that galaxy's own posterior covariance (Cholesky), so a
    correctly-shaped posterior gives the identity. Off-diagonal structure
    there means the reported correlation is wrong, which neither the
    per-parameter rank histograms nor TARP's single scalar would reveal.

    The whitening step assumes the posterior is roughly Gaussian. Use
    `plot_pairwise_rank_calibration` for the distribution-free version.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        names: Feature names, one per parameter.
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, dict with the mean posterior correlation, the
        whitened-residual correlation, its largest off-diagonal
        magnitude, and the whitened per-parameter variances).
    """
    n_gal, _, n_par = samples.shape
    mean = samples.mean(axis=1)
    cov = np.stack([np.cov(s, rowvar=False) for s in samples])

    # Reason: a jitter keeps the Cholesky well posed for parameters the
    # posterior has left essentially at the prior, where draws can be
    # near-degenerate.
    eps = 1e-10 * np.trace(cov, axis1=1, axis2=2)[:, None, None]
    chol = np.linalg.cholesky(cov + eps * np.eye(n_par))
    resid = (truth - mean)[..., None]
    whitened = np.linalg.solve(chol, resid)[..., 0]

    post_corr = np.mean(
        cov / np.sqrt(np.einsum('gii,gjj->gij', cov, cov)), axis=0)
    white_corr = np.corrcoef(whitened, rowvar=False)
    white_var = whitened.var(axis=0)

    off = white_corr - np.eye(n_par)
    max_off = float(np.abs(off).max())

    fig, axes = plt.subplots(1, 2, figsize=(7.2 * 2, 6.6))
    ticks = [tex(n) for n in names]
    for ax, mat, title in (
        (axes[0], post_corr, 'mean posterior correlation'),
        (axes[1], white_corr, 'whitened-residual correlation'),
    ):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(n_par))
        ax.set_yticks(range(n_par))
        ax.set_xticklabels(ticks, rotation=90)
        ax.set_yticklabels(ticks)
        ax.set_title(title)
        ax.grid(False)
        for a in range(n_par):
            for b in range(n_par):
                if abs(mat[a, b]) > 0.25 and a != b:
                    ax.text(b, a, f'{mat[a, b]:.2f}', ha='center',
                            va='center', color='k')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')

    return fig, dict(
        posterior_corr=post_corr, whitened_corr=white_corr,
        max_offdiag=max_off, whitened_var=white_var)


def plot_pairwise_rank_calibration(
    samples: np.ndarray, truth: np.ndarray, names: list[str],
    n_bins: int = 20, n_random: int = 200, seed: int = 0,
    out_path: str | None = None,
):
    """
    Rank calibration along parameter *combinations*, not single axes.

    If the joint posterior is calibrated then the rank of the truth is
    uniform along **any** fixed direction in parameter space, not just
    along the coordinate axes. Testing the difference direction
    `theta_i - theta_j` is what catches a mis-stated correlation: getting
    the correlation too strong squeezes that direction and turns its rank
    histogram U-shaped, while leaving both 1-D marginals untouched.

    The heatmap shows reduced chi-square per pair, with difference
    directions below the diagonal and sum directions above it, and the
    single-parameter values on the diagonal. A set of random directions
    is also scanned, as a check that no combination is badly off.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        names: Feature names, one per parameter.
        n_bins: Number of rank-histogram bins.
        n_random: Number of random unit directions to scan.
        seed: Seed for the random directions.
        out_path: If given, save the figure here.

    Returns:
        Tuple of (figure, dict with the per-pair chi-square matrix, the
        worst pair, and the random-direction summary).
    """
    n_gal, n_draw, n_par = samples.shape
    mat = np.zeros((n_par, n_par))

    for i in range(n_par):
        mat[i, i] = _rank_chi2(samples[:, :, i], truth[:, i], n_bins)
        for j in range(i):
            mat[i, j] = _rank_chi2(
                samples[:, :, i] - samples[:, :, j],
                truth[:, i] - truth[:, j], n_bins)
            mat[j, i] = _rank_chi2(
                samples[:, :, i] + samples[:, :, j],
                truth[:, i] + truth[:, j], n_bins)

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_random, n_par))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    rand_chi2 = np.array([
        _rank_chi2(samples @ u, truth @ u, n_bins) for u in dirs])

    worst = np.unravel_index(np.argmax(mat), mat.shape)
    worst_pair = (names[worst[0]], names[worst[1]], float(mat[worst]))

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    im = ax.imshow(mat, cmap='viridis', vmin=0,
                   vmax=max(3.0, float(mat.max())))
    ticks = [tex(n) for n in names]
    ax.set_xticks(range(n_par))
    ax.set_yticks(range(n_par))
    ax.set_xticklabels(ticks, rotation=90)
    ax.set_yticklabels(ticks)
    ax.set_title(r'rank $\chi^2_\nu$: difference (below), sum (above)')
    ax.grid(False)
    for a in range(n_par):
        for b in range(n_par):
            ax.text(b, a, f'{mat[a, b]:.1f}', ha='center', va='center',
                    color='w' if mat[a, b] < 0.6 * mat.max() else 'k')
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches='tight')

    return fig, dict(
        chi2_matrix=mat, worst_pair=worst_pair,
        random_median=float(np.median(rand_chi2)),
        random_max=float(rand_chi2.max()),
        random_frac_above_2=float(np.mean(rand_chi2 > 2.0)))


def mira_scores(
    samples: np.ndarray, truth: np.ndarray, num_runs: int = 100,
    seed: int = 0,
) -> dict:
    """
    MIRA score for the posterior, against three reference posteriors.

    MIRA is a calibration score built like TARP: it draws a random
    centre, takes the radius out to a random posterior draw, and asks
    whether the truth falls inside as often as the posterior's own draws
    do. Higher is better, bounded above by 1.

    It therefore measures calibration, **not** informativeness. A
    posterior that simply returns the prior is perfectly calibrated and
    scores just as well as a sharp one - which is exactly why the prior
    baseline is computed here as a reference rather than as a floor. Read
    MIRA together with the `pred_vs_true` widths: MIRA says whether the
    error bars are honest, the widths say whether they are useful.

    The two miscalibrated controls bracket the model: broadening and
    sharpening the posterior about its own median should both push the
    score down, and how far is the scale on which to read the model's
    own number.

    Args:
        samples: Posterior samples, shape (n_galaxies, n_draws, n_params).
        truth: True parameters, shape (n_galaxies, n_params).
        num_runs: Monte Carlo replications inside MIRA.
        seed: Seed for the reference posteriors.

    Returns:
        Dict mapping label to (score, standard deviation).
    """
    from mira_score import mira

    rng = np.random.default_rng(seed)
    n_gal, n_draw, _ = samples.shape

    # A posterior that ignored the data entirely: draw each galaxy's
    # "posterior" from the marginal distribution of true parameters.
    # Calibrated but vacuous, so it scores well - see the docstring.
    idx = rng.integers(0, n_gal, size=(n_gal, n_draw))
    prior_like = truth[idx]

    # Miscalibrated controls: same centre, wrong spread, both directions.
    median = np.median(samples, axis=1, keepdims=True)
    broadened = median + 3.0 * (samples - median)
    sharpened = median + (samples - median) / 3.0

    truth_t = torch.tensor(truth, dtype=torch.float32)
    out = {}
    for name, post in (('model', samples),
                       ('model x3 broader', broadened),
                       ('model /3 sharper', sharpened),
                       ('prior baseline', prior_like)):
        score, std = mira(
            truth_t,
            torch.tensor(post, dtype=torch.float32).unsqueeze(0),
            num_runs=num_runs, norm=True, disable_tqdm=True)
        out[name] = (float(score), float(std))
    return out
