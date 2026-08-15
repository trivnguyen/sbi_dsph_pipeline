"""Round r (>= 1): draw a TSNPE-truncated proposal from round r-1's model
and the registered target, then simulate it with Agama.

Requires register_run.py to have already run for this run_dir.

Usage
-----
python simulate_round.py --config configs/debug.py \\
    --config.round=1 --config.proposal.n_sims=1000
"""

import json
import os
import sys
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# Make single-threaded OpenMP the process-wide default, so each of the
# n_jobs worker processes runs one thread instead of one per core. Without
# it the pool oversubscribes by n_workers-fold, which is slower than
# running serially. Must be set before the import chain that reaches
# agama, i.e. before tsnpe.sims below.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import corner
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from absl import flags
from ml_collections import ConfigDict, config_flags

sys.path.insert(0, os.path.dirname(__file__))
from tsnpe.state import RunState
from tsnpe.target import TargetData
from tsnpe import prior
from tsnpe.proposal import sample_tsnpe_proposal
from tsnpe.model_io import build_npe
from tsnpe.sims import run_simulation_batch, write_graph_dataset


def resolve_truths(target_config):
    """Order `config.target.true_params` into `prior.ALL_PARAM_NAMES` order.

    Keyed by name rather than taken positionally on purpose: a truth list
    silently written in the wrong order would mark the wrong crosshairs on
    every corner plot, and nothing downstream would notice.

    Args:
        target_config: `config.target`. `true_params` is optional -- a real
            observation has no truth, and None means no crosshairs.

    Returns:
        A length-8 list in `prior.ALL_PARAM_NAMES` order, or None.

    Raises:
        ValueError: If `true_params` names a parameter that is not in
            `prior.ALL_PARAM_NAMES`, or omits one that is.
    """
    true_params = target_config.get('true_params')
    if not true_params:
        return None

    true_params = dict(true_params)
    unknown = sorted(set(true_params) - set(prior.ALL_PARAM_NAMES))
    if unknown:
        raise ValueError(
            f'config.target.true_params has unknown parameter(s) {unknown}; '
            f'expected names from {prior.ALL_PARAM_NAMES}')
    missing = [n for n in prior.ALL_PARAM_NAMES if n not in true_params]
    if missing:
        raise ValueError(
            f'config.target.true_params is missing {missing}; give every '
            'parameter or leave the whole field unset')

    return [float(true_params[name]) for name in prior.ALL_PARAM_NAMES]


def plot_corner(samples, labels, save_path, title, color='steelblue',
                truths=None, num_max_samples=10000):
    """Save a corner plot of samples, physical units.

    Args:
        samples: (N, D) ndarray.
        labels: Length-D list of parameter names for the plot axes.
        save_path: Destination image path.
        title: Plot title (N is appended automatically).
        color: corner.corner's `color`.
        truths: Length-D true values to mark, or None. Use
            `resolve_truths` rather than building this by hand -- it has
            to be in the same column order as `samples`.
        num_max_samples: Draws are subsampled to this before plotting.
    """
    # downsample for speed: corner.corner is O(N^2) in the number of samples.
    if len(samples) > num_max_samples:
        idx = np.random.choice(len(samples), num_max_samples, replace=False)
        samples = samples[idx]

    fig = corner.corner(
        samples, labels=labels, show_titles=True, title_fmt='.2f',
        title_kwargs={'fontsize': 10}, quantiles=[0.16, 0.5, 0.84],
        label_kwargs={'fontsize': 11}, color=color,
        hist_kwargs={'density': True}, plot_density=True,
        truths=truths, truth_color='crimson',
    )
    fig.suptitle(f'{title} (N={len(samples)})', y=1.01, fontsize=13)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved corner plot -> {save_path}')


PROPOSAL_FILES = ('proposal_phys.npy', 'posterior_phys.npy',
                  'diagnostics.json', 'proposal_settings.json')


def proposal_settings(config, prev_checkpoint, prior_dict):
    """Everything that determines the proposal, for the resume check."""
    return {
        'proposal': dict(config.proposal),
        'prior': prior_dict,
        'checkpoint': str(prev_checkpoint),
        'seed': int(config.seed),
        'round': int(config.round),
    }


def load_cached_proposal(round_dir, settings):
    """Reuse a proposal a previous attempt already drew, if it still fits.

    Drawing the proposal costs a checkpoint load, a tau calibration over
    n_post_samples, and the sampling loop itself; the simulation that
    follows is far longer and far more likely to be what hit the wall
    clock. Re-running the whole step to get back to where it crashed is
    pure waste, so the round's own files are reused when they exist.

    Guarded on `proposal_settings`: a cache drawn under a different
    n_sims, epsilon, sampling_mode or checkpoint describes a different
    distribution, and silently simulating it would be worse than redoing
    the work. Delete the round's .npy files to force a fresh draw.

    Args:
        round_dir: The round's directory.
        settings: `proposal_settings` for the run about to happen.

    Returns:
        (proposal_phys, diagnostics, posterior_phys), or None to draw fresh.
    """
    if not all((round_dir / name).exists() for name in PROPOSAL_FILES):
        return None

    cached = json.loads((round_dir / 'proposal_settings.json').read_text())
    if cached != settings:
        changed = sorted(
            k for k in set(cached) | set(settings)
            if cached.get(k) != settings.get(k))
        print(f'[Proposal] Ignoring the cached proposal in {round_dir}: '
              f'{changed} changed since it was drawn.')
        return None

    proposal_phys = np.load(round_dir / 'proposal_phys.npy')
    posterior_phys = np.load(round_dir / 'posterior_phys.npy')
    diagnostics = json.loads((round_dir / 'diagnostics.json').read_text())
    print(f'[Proposal] Reusing the proposal already drawn in {round_dir} '
          f'({len(proposal_phys):,} draws); skipping the sampling step.')
    return proposal_phys, diagnostics, posterior_phys


def main(config):
    """Simulate round config.round's proposal draws.

    Args:
        config: Pipeline config (see configs/debug.py), with
            `config.round` set to the round to simulate (>= 1).
    """
    r = config.round
    if r < 1:
        raise ValueError(
            f'config.round must be >= 1 (got {r}); round 0 is the '
            'pretrained base model, see register_run.py.')

    state = RunState.load(config.run_dir)

    if state.has_round_field(r, 'data_path'):
        print(f'[Round {r}] Already simulated '
              f'({state.data_path(r)}). Nothing to do.')
        return

    # Round-specific children of the run seed, rather than
    # `config.seed + r`: that collides across runs (seed 0 round 1 is
    # seed 1 round 0) and reuses one stream for both consumers.
    global_seq, star_seq = np.random.SeedSequence([config.seed, r]).spawn(2)
    # seed_everything covers random/numpy/torch; generate_state is uint32,
    # which is exactly its allowed range.
    pl.seed_everything(int(global_seq.generate_state(1)[0]))
    star_rng = np.random.default_rng(star_seq)

    round_dir = state.run_dir / f'round_{r}'
    round_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Round {r}: simulate (n_sims={config.proposal.n_sims}) ===')

    target = TargetData.load(state.target_npz_path())
    norm_dict = json.loads(state.norm_dict_path().read_text())
    model_config = ConfigDict(json.loads(state.model_config_path().read_text()))
    pre_transforms_config = json.loads(state.pre_transforms_config_path().read_text())
    run_prior = prior.Prior.from_dict(
        json.loads(state.prior_config_path().read_text()))
    print(f'[Prior] {run_prior}')
    prev_checkpoint = state.checkpoint_path(r - 1)

    settings = proposal_settings(config, prev_checkpoint, run_prior.to_dict())
    cached = load_cached_proposal(round_dir, settings)

    if cached is not None:
        proposal_phys, diagnostics, posterior_phys = cached
    else:
        print(f'[Model] Warm-starting from round {r - 1}: {prev_checkpoint}')
        # pre_transforms=None: sample_tsnpe_proposal always builds its own
        # observation-only pre_transforms and passes it explicitly, so the
        # model never falls back to a self.pre_transforms of its own.
        model = build_npe(model_config, None, norm_dict)
        # weights_only=False: round >= 2 loads a full Lightning checkpoint,
        # not just a state_dict; safe since it's always our own output.
        checkpoint = torch.load(
            prev_checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        print('[Proposal] Sampling TSNPE-truncated proposal...')
        proposal_phys, diagnostics, posterior_phys = sample_tsnpe_proposal(
            model, run_prior, target, norm_dict, pre_transforms_config,
            return_posterior=True, **config.proposal)
        print(f'  proposal_phys : {proposal_phys.shape}')
        print(f'  diagnostics   : {diagnostics}')

        # Settings last: it is what load_cached_proposal keys on, so it
        # must not appear before the arrays it describes are on disk.
        np.save(round_dir / 'proposal_phys.npy', proposal_phys)
        np.save(round_dir / 'posterior_phys.npy', posterior_phys)
        with open(round_dir / 'diagnostics.json', 'w') as f:
            json.dump(diagnostics, f, indent=2)
        with open(round_dir / 'proposal_settings.json', 'w') as f:
            json.dump(settings, f, indent=2)

    truths = resolve_truths(config.target)
    if truths is not None:
        print(f'[Plot] Marking truth: '
              f'{dict(zip(prior.ALL_PARAM_NAMES, truths))}')

    plot_corner(
        proposal_phys, prior.ALL_PARAM_NAMES, round_dir / 'proposal_corner.png',
        'Proposal samples', truths=truths)
    plot_corner(
        posterior_phys, prior.ALL_PARAM_NAMES, round_dir / 'posterior_corner.png',
        'Posterior samples', color='darkorange', truths=truths)

    print('[Simulate] Running Agama simulation batch...')
    sim_cfg = config.simulation
    n_stars = star_rng.poisson(
        sim_cfg.n_stars_mean, len(proposal_phys)).tolist()
    theta, posvel_list = run_simulation_batch(
        proposal_phys, n_stars, n_jobs=sim_cfg.n_jobs,
        use_multiprocessing=sim_cfg.use_multiprocessing,
        sample_threads=sim_cfg.sample_threads)

    data_path = round_dir / 'data.hdf5'
    write_graph_dataset(
        str(data_path), theta, posvel_list, prior.ALL_PARAM_NAMES,
        headers={
            'name': f'{target.key}_tsnpe_round{r}', 'round': r,
            'n_sims_requested': config.proposal.n_sims,
            # Carried verbatim rather than cherry-picked: the sampling
            # modes report different fields ('rejection'/'flow' give
            # acceptance_rate/n_accepted, 'sir' gives ess_total/n_total),
            # and naming one of them here KeyErrors under the others --
            # after the whole simulation batch has already run. All of
            # them are scalars, so they store as HDF5 attrs unchanged.
            **diagnostics,
        },
    )
    n_success = len(posvel_list)
    print(f'[Simulate] Successful simulations: {n_success} / {len(proposal_phys)} '
          f'({n_success / len(proposal_phys) * 100:.1f}%)')

    state.register_round(
        r,
        data_path=str(data_path.relative_to(state.run_dir)),
        diagnostics=diagnostics,
        n_sims_requested=config.proposal.n_sims,
        n_sims_successful=n_success,
    )
    print(f'[Round {r}] Wrote dataset -> {data_path}')


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        'config', None, 'Path to an ml_collections Python config file (*.py).',
        lock_config=True)
    FLAGS(sys.argv)
    main(config=FLAGS.config)
