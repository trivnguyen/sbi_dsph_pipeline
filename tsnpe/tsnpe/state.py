"""Run-state manifest for the TSNPE pipeline.

Every script reads and writes ``<run_dir>/state.json`` instead of taking
checkpoint / x_obs paths as ad-hoc CLI flags. All paths inside the manifest
are stored *relative to run_dir*, and the files they point to are hard
copies (never references to external catalogs or wandb caches) — so a run
directory is self-contained and can be moved or copied elsewhere without
breaking anything.

Round 0 (the "base" model) is special: its checkpoint, norm_dict, and model
architecture are fixed for the entire run and never recomputed by later
rounds. Rounds 1..N only ever add a new checkpoint fine-tuned from the
previous round.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

STATE_FILENAME = 'state.json'
SEED_OFFSET = 100000  # avoiding seed collision between sim and train round

def sha256_file(path: Path, chunk_size: int = 2 ** 20) -> str:
    """Compute the sha256 hex digest of a file.

    Args:
        path: File to hash.
        chunk_size: Bytes read per chunk.

    Returns:
        Hex-encoded sha256 digest.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class RunState:
    """In-memory view of a run's state.json.

    Attributes:
        run_dir: Root directory for the run.
        seed: Base random seed for the run, fixed at creation.
        target: Registered observation record, or None if not yet registered.
        base: Round-0 (pretrained) model record, or None if not yet registered.
        rounds: Mapping from round index (>=1) to that round's record.
    """
    run_dir: Path
    seed: int
    target: Optional[dict[str, Any]] = None
    base: Optional[dict[str, Any]] = None
    rounds: dict[int, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, run_dir: str | Path) -> 'RunState':
        """Load state.json from run_dir.

        Args:
            run_dir: Root directory for the run.

        Returns:
            The loaded RunState.

        Raises:
            FileNotFoundError: If state.json does not exist yet.
        """
        run_dir = Path(run_dir)
        path = run_dir / STATE_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No {STATE_FILENAME} in {run_dir}. Run register_run.py first.")
        with open(path) as f:
            raw = json.load(f)
        rounds = {int(k): v for k, v in raw.get('rounds', {}).items()}
        return cls(
            run_dir=run_dir, seed=raw['seed'],
            target=raw.get('target'), base=raw.get('base'), rounds=rounds)

    @classmethod
    def load_or_create(cls, run_dir: str | Path, seed: int) -> 'RunState':
        """Load run_dir's state.json, creating a fresh one if it's missing.

        Args:
            run_dir: Root directory for the run (created if missing).
            seed: Base random seed. Must match the recorded seed if the run
                already exists.

        Returns:
            The loaded or newly created RunState.

        Raises:
            ValueError: If `seed` conflicts with a previously recorded seed.
        """
        run_dir = Path(run_dir)
        path = run_dir / STATE_FILENAME
        if path.exists():
            state = cls.load(run_dir)
            if state.seed != seed:
                raise ValueError(
                    f"config.seed={seed} conflicts with the seed already "
                    f"recorded in {path} ({state.seed}). Omit config.seed "
                    "to reuse it, or point config.run_dir at a fresh "
                    "directory to start a new run.")
            return state
        run_dir.mkdir(parents=True, exist_ok=True)
        state = cls(run_dir=run_dir, seed=seed)
        state.save()
        return state

    def save(self) -> None:
        """Persist this state to `<run_dir>/state.json` (atomically)."""
        path = self.run_dir / STATE_FILENAME
        raw = {
            'seed': self.seed,
            'target': self.target,
            'base': self.base,
            'rounds': {str(k): v for k, v in sorted(self.rounds.items())},
        }
        tmp_path = path.with_suffix('.json.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(raw, f, indent=2)
        tmp_path.replace(path)

    def resolve(self, relative_path: str) -> Path:
        """Resolve a path stored in state.json against run_dir."""
        return self.run_dir / relative_path

    # ------------------------------------------------------------------
    # Target (observational data)
    # ------------------------------------------------------------------
    def register_target(self, npz_path: Path, **provenance: Any) -> None:
        """Record the target snapshot at `npz_path` (already copied into run_dir).

        Args:
            npz_path: Path to the copied target .npz file, inside run_dir.
            **provenance: Extra fields to store verbatim (e.g. catalog_path,
                key) for traceability back to the source catalog.
        """
        self.target = {
            'npz_path': str(npz_path.relative_to(self.run_dir)),
            'sha256': sha256_file(npz_path),
            **provenance,
        }
        self.save()

    def require_target(self) -> dict[str, Any]:
        """Return the registered target record.

        Raises:
            RuntimeError: If no target has been registered yet.
        """
        if self.target is None:
            raise RuntimeError(
                f"No target registered in {self.run_dir}. "
                "Run register_run.py first.")
        return self.target

    def target_npz_path(self) -> Path:
        """Absolute path to the registered target .npz snapshot."""
        return self.resolve(self.require_target()['npz_path'])

    # ------------------------------------------------------------------
    # Base (round-0) model
    # ------------------------------------------------------------------
    def register_base(
        self, checkpoint_path: Path, norm_dict_path: Path,
        model_config_path: Path, pre_transforms_config_path: Path,
        **provenance: Any,
    ) -> None:
        """Record the round-0 base model (already copied into run_dir).

        Args:
            checkpoint_path: Path to the copied checkpoint, inside run_dir.
            norm_dict_path: Path to the extracted norm_dict.json, inside run_dir.
            model_config_path: Path to the extracted model_config.json, inside run_dir.
            pre_transforms_config_path: Path to the extracted
                pre_transforms_config.json, inside run_dir.
            **provenance: Extra fields to store verbatim (e.g. wandb_run_path)
                for traceability back to the source run.
        """
        self.base = {
            'checkpoint_path': str(checkpoint_path.relative_to(self.run_dir)),
            'norm_dict_path': str(norm_dict_path.relative_to(self.run_dir)),
            'model_config_path': str(model_config_path.relative_to(self.run_dir)),
            'pre_transforms_config_path':
                str(pre_transforms_config_path.relative_to(self.run_dir)),
            **provenance,
        }
        self.save()

    def require_base(self, field_name: Optional[str] = None) -> Any:
        """Return the base-model record, or one field of it.

        Args:
            field_name: If given, return only this field's value.

        Raises:
            RuntimeError: If no base model is registered yet.
        """
        if self.base is None:
            raise RuntimeError(
                f"No base (round-0) model registered in {self.run_dir}. "
                "Run register_run.py first.")
        if field_name is None:
            return self.base
        return self.base[field_name]

    def norm_dict_path(self) -> Path:
        """Absolute path to the run's (fixed, round-0) norm_dict.json."""
        return self.resolve(self.require_base('norm_dict_path'))

    def model_config_path(self) -> Path:
        """Absolute path to the run's (fixed, round-0) model_config.json."""
        return self.resolve(self.require_base('model_config_path'))

    def pre_transforms_config_path(self) -> Path:
        """Absolute path to the run's (fixed, round-0) pre_transforms_config.json."""
        return self.resolve(self.require_base('pre_transforms_config_path'))

    # ------------------------------------------------------------------
    # Rounds (>= 1)
    # ------------------------------------------------------------------
    def register_round(self, round_idx: int, **fields: Any) -> None:
        """Merge `fields` into round `round_idx`'s record and save.

        Args:
            round_idx: Round index (>= 1).
            **fields: Fields to set, e.g. data_path, checkpoint_path,
                diagnostics, wandb_run_id.
        """
        record = self.rounds.setdefault(round_idx, {})
        record.update(fields)
        self.save()

    def require_round(self, round_idx: int, field_name: Optional[str] = None) -> Any:
        """Return round `round_idx`'s record, or one field of it.

        Args:
            round_idx: Round index to look up.
            field_name: If given, return only this field's value.

        Returns:
            The round record dict, or the value of `field_name` within it.

        Raises:
            RuntimeError: If the round (or the requested field) isn't
                recorded yet — the previous step hasn't finished.
        """
        record = self.rounds.get(round_idx)
        if record is None:
            raise RuntimeError(
                f"Round {round_idx} has no record in "
                f"{self.run_dir / STATE_FILENAME}.")
        if field_name is None:
            return record
        if field_name not in record:
            raise RuntimeError(
                f"Round {round_idx} record in "
                f"{self.run_dir / STATE_FILENAME} is missing '{field_name}'. "
                f"Has round {round_idx}'s earlier step finished?")
        return record[field_name]

    def checkpoint_path(self, round_idx: int) -> Path:
        """Absolute path to round `round_idx`'s checkpoint (0 = base model)."""
        if round_idx == 0:
            return self.resolve(self.require_base('checkpoint_path'))
        return self.resolve(self.require_round(round_idx, 'checkpoint_path'))

    def data_path(self, round_idx: int) -> Path:
        """Absolute path to round `round_idx`'s simulated dataset."""
        return self.resolve(self.require_round(round_idx, 'data_path'))

    def has_round_field(self, round_idx: int, field_name: str) -> bool:
        """Return whether round `round_idx` already has `field_name` recorded."""
        record = self.rounds.get(round_idx)
        return record is not None and field_name in record

    def latest_trained_round(self) -> int:
        """Return the highest round index with a checkpoint (0 if only base)."""
        trained = [r for r, rec in self.rounds.items() if 'checkpoint_path' in rec]
        return max(trained) if trained else 0
