"""Round r (>= 1): fine-tune round r-1's checkpoint on round r's freshly
simulated data.

Trains on only round r's own dataset, warm-started from round r-1's
weights with a fresh optimizer. Normalization dict and model architecture
are fixed at round 0 (register_run.py), never recomputed here.

Requires simulate_round.py to have already produced round_<r>/data.hdf5.

Usage
-----
python train_round.py --config configs/debug.py --config.round=1
"""

import json
import os
os.environ['WANDB_DATA_DIR'] = '/scratch/tvnguyen/wandb_data'

import sys
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from pathlib import Path

import wandb
import pytorch_lightning as pl
from absl import flags
from ml_collections import ConfigDict, config_flags
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.model_summary import summarize

sys.path.insert(0, os.path.dirname(__file__))
from tsnpe.state import RunState, SEED_OFFSET
from tsnpe.model_io import build_npe
from tsnpe import prior

from jgnn import datasets, training
from jgnn.transforms import build_transformation
from jgnn.callbacks.visualization import NPEVisualizationCallback


def prepare_data(config, data_path, norm_dict, pre_transforms_config):
    """Build train/val dataloaders from one round's simulated Cartesian dataset.

    Args:
        config: Pipeline config (see configs/debug.py).
        data_path: Path to the round's data.hdf5 (tsnpe/sims.py's output).
        norm_dict: Fixed normalization dict (reused, never recomputed).
        pre_transforms_config: The round-0 model's pre_transforms config
            (state.pre_transforms_config_path()).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    node_feats, graph_feats, _ = datasets.read_graph_dataset(str(data_path), concat=True)
    seed_data = config.seed + config.round + SEED_OFFSET

    train_loader, val_loader, _ = datasets.cartesian.prepare_dataloaders(
        node_feats, graph_feats, prior.PARAM_NAMES,
        cond_labels=[prior.CONDITIONING_NAME],
        train_batch_size=config.training.train_batch_size,
        eval_batch_size=config.training.eval_batch_size,
        train_frac=config.training.train_frac,
        num_workers=config.training.num_workers,
        seed=seed_data,
        norm_dict=norm_dict,
        pre_transform_kwargs=pre_transforms_config,
    )
    return train_loader, val_loader


def create_callbacks(config: ConfigDict) -> list:
    """Create PyTorch Lightning callbacks, including NPE-specific visualization.

    Args:
        config: Configuration dictionary

    Returns:
        List of callback instances
    """
    callbacks = training.create_base_callbacks(config)

    if config.get('enable_visualization_callback', False):
        print("[Callbacks] Adding NPE Visualization Callback")
        callbacks.append(
            NPEVisualizationCallback(
                plot_every_n_epochs=config.visualization.get('plot_every_n_epochs', 1),
                n_posterior_samples=config.visualization.get('n_posterior_samples', 1000),
                n_val_samples=config.visualization.get('n_val_samples', 100),
                plot_median_v_true=config.visualization.get('plot_median_v_true', True),
                plot_tarp=config.visualization.get('plot_tarp', True),
                plot_rank=config.visualization.get('plot_rank', True),
                use_default_mplstyle=config.visualization.get('use_default_mplstyle', True),
            )
        )
    return callbacks


def main(config):
    """Fine-tune round config.round's model on its freshly simulated data."""
    r = config.round
    if r < 1:
        raise ValueError(
            f'config.round must be >= 1 (got {r}); round 0 is the '
            'pretrained base model, see register_run.py.')

    state = RunState.load(config.run_dir)

    if state.has_round_field(r, 'checkpoint_path'):
        print(f'[Round {r}] Already trained '
              f'({state.checkpoint_path(r)}). Nothing to do.')
        return

    round_dir = state.run_dir / f'round_{r}'

    # Reuse a previously-recorded wandb run id for this round, if any, so a
    # retry (e.g. after a crash before registration, below) resumes the
    # same wandb run instead of leaking an orphaned new one every attempt.
    # workdir is set to round_dir (not a fixed/shared path) so checkpoints
    # and wandb's own local run files both stay under round_dir, keeping
    # the run directory self-contained (see tsnpe/state.py).
    existing_run_id = state.rounds.get(r, {}).get('wandb_run_id')
    if existing_run_id is not None:
        config.training.id = existing_run_id
    config.training.workdir = str(round_dir)

    wandb_logger, project_dir = training.create_wandb_logger(config.training, tag=f'tsnpe-round{r}')
    # Persist the run id immediately - before training starts - so a crash
    # anywhere after this point (mid-training, or even after training
    # finishes but before this function's own registration below) can be
    # resumed via existing_run_id above instead of orphaning a new run.
    state.register_round(r, wandb_run_id=wandb_logger.experiment.id)

    expected_ckpt = project_dir / 'checkpoints' / 'last.ckpt'
    if expected_ckpt.exists():
        print(f'[Round {r}] Found a completed checkpoint from a previous '
              f'attempt that crashed before registering: {expected_ckpt}')
        state.register_round(
            r, checkpoint_path=str(expected_ckpt.relative_to(state.run_dir)))
        print(f'[Round {r}] Registered checkpoint -> {expected_ckpt}')
        wandb.finish()
        return

    data_path = state.data_path(r)
    prev_checkpoint = state.checkpoint_path(r - 1)
    norm_dict = json.loads(state.norm_dict_path().read_text())
    model_config = ConfigDict(json.loads(state.model_config_path().read_text()))
    pre_transforms_config = json.loads(state.pre_transforms_config_path().read_text())

    print(f'=== Round {r}: train ===')
    print(f'[Data] Loading {data_path}')
    train_loader, val_loader = prepare_data(config, data_path, norm_dict, pre_transforms_config)
    print(f'[Data] Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    print('[Transforms] Building pre-transforms...')
    pre_transforms = build_transformation(norm_dict=norm_dict, **pre_transforms_config)

    print('[Model] Creating NPE model...')
    model = build_npe(
        model_config, pre_transforms, norm_dict,
        optimizer_args=config.training.optimizer,
        scheduler_args=config.training.scheduler,
    )
    summarize(model, max_depth=3)
    training.report_param_counts(model)

    wandb_logger.watch(model, log='all', log_freq=1000, log_graph=False)

    callbacks = create_callbacks(config.training)
    trainer = training.build_trainer(
        config.training, project_dir, callbacks, wandb_logger, num_sanity_val_steps=0)

    print(f'[Model] Warm-starting from round {r - 1}: {prev_checkpoint}')
    seed_training = config.seed + r + SEED_OFFSET * 2
    pl.seed_everything(seed_training)
    training.fit(
        trainer, model, train_loader, val_loader,
        checkpoint_path=str(prev_checkpoint), reset_optimizer=True)

    # With a WandbLogger attached, ModelCheckpoint writes under
    # round_dir/<wandb_project>/<run_id>/checkpoints/, not the
    # logger-less default of round_dir/lightning_logs/checkpoints/ - ask
    # the callback directly instead of assuming a fixed path.
    last_ckpt_cb = next(
        cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.save_last)
    checkpoint_path = Path(last_ckpt_cb.last_model_path)
    if not checkpoint_path.exists():
        raise RuntimeError(
            f'Expected checkpoint not found at {checkpoint_path}; '
            'training may have failed before saving.')

    state.register_round(
        r,
        checkpoint_path=str(checkpoint_path.relative_to(state.run_dir)),
        wandb_run_id=wandb_logger.experiment.id,
    )
    print(f'[Round {r}] Trained checkpoint -> {checkpoint_path}')

    wandb.finish()


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        'config', None, 'Path to an ml_collections Python config file (*.py).',
        lock_config=True)
    FLAGS(sys.argv)
    main(config=FLAGS.config)
