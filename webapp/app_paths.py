"""Where this app imports `tsnpe`, `jgnn` and `dsph_analysis` from.

Two mutually exclusive modes, and the distinction matters enough to
live in one file rather than be repeated in each module that needs it:

Bundle
    `package.py` vendors the three packages next to `app.py`. Those
    frozen copies are the *only* ones consulted - a bundle built on the
    machine that also holds the repo must not quietly import the repo's
    code, or testing the bundle locally would exercise the wrong
    version of it.

Repo
    This directory lives inside `sbi_dsph_pipeline`, so the pipeline's
    `tsnpe` package is simply `../tsnpe` and moves with the checkout.
    `dsph_analysis` is outside it and is located by absolute path;
    that default matches this machine, and the environment variables
    are named exactly as in `npe_inference/npe_infer/paths.py`, so a
    checkout that moves costs an export rather than an edit here.

`setup()` is idempotent and safe to call from any module at import
time - whichever of `inference` and `user_catalog` is imported first
does the work.

`load_registry()` is the same idea applied to *models* rather than
packages: in repo mode the set of switchable checkpoints is read from
`npe_inference/configs/models.py` instead of being restated here,
because a second list of radius-unit conventions that could drift from
the first is exactly the failure `tsnpe/prior.py` warns about. A
bundle carries one model and no registry.
"""

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

SBI_DSPH_PIPELINE_DIR = Path(os.environ.get(
    'SBI_DSPH_PIPELINE_DIR', str(APP_DIR.parent)))
MY_MODULES_DIR = Path(os.environ.get(
    'MY_MODULES_DIR', '/home/tvnguyen/my_modules'))

# The project holding configs/models.py, the registry of checkpoints
# this app may be pointed at. Repo mode only.
NPE_INFERENCE_DIR = Path(os.environ.get(
    'NPE_INFERENCE_DIR',
    str(SBI_DSPH_PIPELINE_DIR.parent / 'npe_inference')))

# Trained runs live at <MODEL_WORKDIR>/<project>/<run_id>/. Same
# variable name npe_infer.paths uses, so one export covers both.
MODEL_WORKDIR = Path(os.environ.get(
    'NPE_MODEL_WORKDIR', '/scratch/tvnguyen/trained_models/dsph_npe'))

# The presence of a vendored package next to app.py is what tells a
# bundle from a checkout.
IS_BUNDLE = (APP_DIR / 'tsnpe').is_dir()


def import_path() -> tuple[Path, ...]:
    """The directories to put on `sys.path`, highest priority first.

    Returns:
        `(APP_DIR,)` in a bundle; the repo's `tsnpe` directory and the
        `dsph_analysis` parent otherwise.
    """
    if IS_BUNDLE:
        return (APP_DIR,)
    return (SBI_DSPH_PIPELINE_DIR / 'tsnpe', MY_MODULES_DIR)


def setup() -> None:
    """Make `tsnpe`, `jgnn` and `dsph_analysis` importable. Idempotent."""
    for directory in reversed(import_path()):
        entry = str(directory)
        if directory.is_dir() and entry not in sys.path:
            sys.path.insert(0, entry)


def load_registry() -> dict[str, dict]:
    """The checkpoints this server may switch between, by CLI name.

    Read from `npe_inference/configs/models.py` rather than restated
    here: that file is where a model's radius convention is asserted,
    and it explains at length why the convention is a deliberate
    per-model declaration and not something to be inferred. Two copies
    of that list could disagree, and the symptom - `dm_log_rdm` off by
    a factor of r_star - looks like a plausible number rather than an
    error.

    Entries whose checkpoint is not on this filesystem are dropped, so
    a registry naming runs that were never copied here still yields a
    usable subset instead of failing at startup.

    Returns:
        `{name: {name, model_dir, checkpoint, radius_units, note}}`,
        empty in a bundle (which carries exactly one model, described
        by its own `model/model_spec.json`) or if the registry is not
        importable from this machine.
    """
    if IS_BUNDLE:
        return {}
    entry = str(NPE_INFERENCE_DIR)
    if not (NPE_INFERENCE_DIR / 'configs' / 'models.py').is_file():
        return {}
    if entry not in sys.path:
        sys.path.insert(0, entry)
    try:
        from configs.models import MODELS
    except ImportError:
        return {}

    registry = {}
    for name, spec in MODELS.items():
        model_dir = MODEL_WORKDIR / spec.project / spec.run_id
        checkpoints = model_dir / 'checkpoints'
        if not (checkpoints / spec.checkpoint).is_file():
            continue
        registry[name] = dict(
            name=name, model_dir=checkpoints,
            checkpoint=spec.checkpoint,
            radius_units=spec.radius_units, note=spec.note)
    return registry
