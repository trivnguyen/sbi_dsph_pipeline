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


def plot_corner(samples, labels, save_path, title, color='steelblue'):
    """Save a corner plot of samples, physical units.

    Args:
        samples: (N, D) ndarray.
        labels: Length-D list of parameter names for the plot axes.
        save_path: Destination image path.
        title: Plot title (N is appended automatically).
        color: corner.corner's `color`.
    """
    fig = corner.corner(
        samples, labels=labels, show_titles=True, title_fmt='.2f',
        title_kwargs={'fontsize': 10}, quantiles=[0.16, 0.5, 0.84],
        label_kwargs={'fontsize': 11}, color=color,
        hist_kwargs={'density': True}, plot_density=True,
    )
    fig.suptitle(f'{title} (N={len(samples)})', y=1.01, fontsize=13)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'[Plot] Saved corner plot -> {save_path}')


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

    pl.seed_everything(config.seed + r)
    np.random.seed(config.seed + r)


    round_dir = state.run_dir / f'round_{r}'
    round_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Round {r}: simulate (n_sims={config.proposal.n_sims}) ===')

    target = TargetData.load(state.target_npz_path())
    norm_dict = json.loads(state.norm_dict_path().read_text())
    model_config = ConfigDict(json.loads(state.model_config_path().read_text()))
    pre_transforms_config = json.loads(state.pre_transforms_config_path().read_text())
    prev_checkpoint = state.checkpoint_path(r - 1)

    print(f'[Model] Warm-starting from round {r - 1}: {prev_checkpoint}')
    # pre_transforms=None: sample_tsnpe_proposal always builds its own
    # observation-only pre_transforms and passes it explicitly, so the
    # model never falls back to a self.pre_transforms of its own.
    model = build_npe(model_config, None, norm_dict)
    # weights_only=False: round >= 2 loads a full Lightning checkpoint, not
    # just a state_dict; safe since it's always our own pipeline's output.
    checkpoint = torch.load(prev_checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print('[Proposal] Sampling TSNPE-truncated proposal...')
    proposal_phys, diagnostics, posterior_phys = sample_tsnpe_proposal(
        model, target, norm_dict, pre_transforms_config,
        return_posterior=True, **config.proposal)
    print(f'  proposal_phys : {proposal_phys.shape}')
    print(f'  diagnostics   : {diagnostics}')

    np.save(round_dir / 'proposal_phys.npy', proposal_phys)
    np.save(round_dir / 'posterior_phys.npy', posterior_phys)
    with open(round_dir / 'diagnostics.json', 'w') as f:
        json.dump(diagnostics, f, indent=2)

    plot_corner(
        proposal_phys, prior.ALL_PARAM_NAMES, round_dir / 'proposal_corner.png',
        'Proposal samples')
    plot_corner(
        posterior_phys, prior.ALL_PARAM_NAMES, round_dir / 'posterior_corner.png',
        'Posterior samples', color='darkorange')

    print('[Simulate] Running Agama simulation batch...')
    sim_cfg = config.simulation
    n_stars = np.random.poisson(
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
            'tau': diagnostics['tau'],
            'acceptance_rate': diagnostics['acceptance_rate'],
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
