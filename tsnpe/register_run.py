"""One-time setup: register a target's observation and the run's round-0
("base") model.

Usage
-----
python register_run.py --config configs/debug.py

Debug mode: set config.pretrained.random_init = True to build a fresh
random model instead (see tsnpe.model_io.debug_model_config /
tsnpe.prior.default_norm_dict). Doesn't require a target either.

The architecture and pre_transforms recipe that trained the checkpoint are
never hand-specified, since a hand-written copy could silently drift from
what the checkpoint actually expects:
  - config.pretrained.local_checkpoint_dir is fully offline: it never
    calls wandb. It looks for config_snapshot.json (written by
    npe/train_npe.py) near the checkpoint file and fails clearly if it's
    missing, rather than silently falling back to the network.
  - Without local_checkpoint_dir, both are read from wandb (that's the
    only other place training logs them).
config.pretrained.pre_transforms_override is the escape hatch for a
checkpoint with neither (trained before pre_transforms was logged either
locally or to wandb).
"""

import json
import os
import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning)

import torch
from absl import flags
from ml_collections import ConfigDict, config_flags

sys.path.insert(0, os.path.dirname(__file__))
from tsnpe.state import RunState
from tsnpe.target import TargetData
from tsnpe import prior
from tsnpe.model_io import build_npe, debug_model_config, debug_pre_transforms_config

from jgnn import utils


def _to_plain_dict(x) -> dict:
    return x.to_dict() if hasattr(x, 'to_dict') else dict(x)


def _norm_dict_from_checkpoint(checkpoint_path: str) -> dict:
    """Read norm_dict straight from a checkpoint's embedded hyperparameters.

    No model needs to be built for this - unlike
    jgnn.utils.load_npe_from_checkpoint, which does (and so needs
    model_config just to read norm_dict, which for wandb-sourced configs
    means a network call this doesn't otherwise need).
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['hyper_parameters']['norm_dict']


def register_target(config, state: RunState) -> None:
    """Register config.target's observational data into config.run_dir."""
    if state.target is not None:
        print(f"[Target] Already registered (key={state.target.get('key')}).")
        return

    print(f"[Target] Loading '{config.target.key}' from {config.target.catalog_path}")
    target = TargetData.from_catalog(config.target)

    target_dir = state.run_dir / 'target'
    target_dir.mkdir(parents=True, exist_ok=True)
    npz_path = target_dir / 'x_obs.npz'
    target.save(npz_path)

    state.register_target(
        npz_path, key=config.target.key,
        catalog_path=config.target.catalog_path,
        n_stars=len(target.ra_deg))
    print(f"[Target] Registered {len(target.ra_deg)} stars -> {npz_path}")


def _resolve_configs(full_config, pretrained_config):
    """Pick model_config/pre_transforms_config out of a full training config."""
    model_config = full_config.model
    pre_transforms_config = (
        pretrained_config.get('pre_transforms_override')
        or full_config.get('pre_transforms'))
    if not pre_transforms_config:
        raise ValueError(
            "No pre_transforms found (locally or on wandb) for this "
            "checkpoint, and config.pretrained.pre_transforms_override "
            "is not set.")
    return model_config, pre_transforms_config


def _find_config_snapshot(checkpoint_path: str, max_levels: int = 3) -> Path | None:
    """Look for config_snapshot.json in the checkpoint's directory, or a few
    parents up - it may sit right next to the checkpoint
    (workdir/<project>/<run_id>/config_snapshot.json, alongside
    checkpoints/) or higher, at the training run's workdir root.
    """
    current = Path(checkpoint_path).resolve().parent
    for _ in range(max_levels):
        candidate = current / 'config_snapshot.json'
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _resolve_checkpoint(pretrained_config):
    """Resolve a checkpoint file + model_config + pre_transforms_config + norm_dict.

    `local_checkpoint_dir` is fully offline: it never calls wandb. It
    requires a config_snapshot.json near the checkpoint file (written by
    npe/train_npe.py) and fails clearly if one isn't found. Without
    `local_checkpoint_dir`, both the checkpoint and its config come from
    wandb. norm_dict always comes straight from the checkpoint file's own
    embedded hyperparameters either way, never wandb.

    Returns:
        (checkpoint_path, model_config, pre_transforms_config, norm_dict, provenance)
    """
    if pretrained_config.get('local_checkpoint_dir'):
        checkpoint_path = utils.fetch_local_checkpoint(
            pretrained_config.local_checkpoint_dir,
            pretrained_config.get('local_checkpoint_filename', 'model.ckpt'))
        norm_dict = _norm_dict_from_checkpoint(checkpoint_path)

        snapshot_path = _find_config_snapshot(checkpoint_path)
        if snapshot_path is None:
            raise FileNotFoundError(
                f"No config_snapshot.json found near {checkpoint_path}. "
                "local_checkpoint_dir never calls wandb, so this can't be "
                "resolved automatically - either make sure the checkpoint "
                "has a snapshot (written by npe/train_npe.py), or unset "
                "local_checkpoint_dir to fetch from wandb instead.")

        full_config = ConfigDict(json.loads(snapshot_path.read_text()))
        model_config, pre_transforms_config = _resolve_configs(full_config, pretrained_config)
        provenance = {
            'source': 'local_offline',
            'local_checkpoint_dir': pretrained_config.local_checkpoint_dir,
        }
        return checkpoint_path, model_config, pre_transforms_config, norm_dict, provenance

    checkpoint_path, full_config = utils.fetch_wandb_checkpoint(
        run_path=pretrained_config.wandb_run_path,
        version=pretrained_config.get('wandb_version', 'best'))
    model_config, pre_transforms_config = _resolve_configs(full_config, pretrained_config)
    norm_dict = _norm_dict_from_checkpoint(checkpoint_path)
    provenance = {'source': 'wandb', 'wandb_run_path': pretrained_config.wandb_run_path}
    return checkpoint_path, model_config, pre_transforms_config, norm_dict, provenance


def register_pretrained(config, state: RunState) -> None:
    """Register config.pretrained's checkpoint as round 0 in config.run_dir."""
    if state.base is not None:
        print(f"[Base] Already registered in {state.run_dir}.")
        return

    round0_dir = state.run_dir / 'round_0'
    round0_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dst = round0_dir / 'model.ckpt'

    if config.pretrained.get('random_init', False):
        print("[Base] random_init=True: using a freshly-initialized debug model.")
        model_config = debug_model_config()
        pre_transforms_config = debug_pre_transforms_config()
        norm_dict = prior.default_norm_dict()
        model = build_npe(model_config, pre_transforms=None, norm_dict=norm_dict)
        torch.save({'state_dict': model.state_dict()}, ckpt_dst)
        provenance = {'source': 'random_init'}
    else:
        print("[Base] Resolving pretrained checkpoint...")
        checkpoint_path, model_config, pre_transforms_config, norm_dict, provenance = \
            _resolve_checkpoint(config.pretrained)
        shutil.copy2(checkpoint_path, ckpt_dst)

    norm_dict_dst = round0_dir / 'norm_dict.json'
    with open(norm_dict_dst, 'w') as f:
        json.dump(norm_dict, f, indent=2)

    model_config_dst = round0_dir / 'model_config.json'
    with open(model_config_dst, 'w') as f:
        json.dump(_to_plain_dict(model_config), f, indent=2)

    pre_transforms_config_dst = round0_dir / 'pre_transforms_config.json'
    with open(pre_transforms_config_dst, 'w') as f:
        json.dump(_to_plain_dict(pre_transforms_config), f, indent=2)

    state.register_base(
        ckpt_dst, norm_dict_dst, model_config_dst, pre_transforms_config_dst,
        **provenance)
    print(f"[Base] Registered round-0 model -> {round0_dir}")


def main(config):
    """Register config.target and config.pretrained into config.run_dir."""
    if config.overwrite and Path(config.run_dir).exists():
        print(f"[Main] Overwriting existing run_dir {config.run_dir}.")
        shutil.rmtree(config.run_dir)

    state = RunState.load_or_create(config.run_dir, config.seed)
    register_target(config, state)
    register_pretrained(config, state)


if __name__ == '__main__':
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        'config', None, 'Path to an ml_collections Python config file (*.py).',
        lock_config=True)
    FLAGS(sys.argv)
    main(config=FLAGS.config)
