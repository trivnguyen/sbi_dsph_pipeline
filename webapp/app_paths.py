"""Where this app imports `tsnpe`, `jgnn` and `dsph_analysis` from.

All three come from `vendor/` - the app's own committed copies - and
from nowhere else. No checkout elsewhere on the machine is consulted,
and neither is an editable install of any of them. Outside `vendor/`
the app depends only on ordinary PyPI packages (torch, numpy, scipy,
astropy, ...), which pin themselves through the environment.

That is deliberate. Those three are research code under active
development, and while the app imported them live, an improvement to
one arrived in the running server the moment it was committed
somewhere else, with nothing recording which version a result came
from. A change to `dsph_analysis`'s Jeans solver did exactly that and
cost a day of debugging an app that had not changed at all.

So a dependency update is an explicit commit: `python vendor.py
--refresh`, review, commit. `python vendor.py` reports what is
vendored against what the live checkouts hold, without the server ever
consulting them.

`setup()` is idempotent and safe to call from any module at import
time - whichever of `inference` and `user_catalog` is imported first
does the work. A bundle built by `package.py` carries the same
`vendor/` tree, so a bundle and a repo-mode server run byte-identical
dependency code.

`load_registry()` is the same idea applied to *models* rather than
packages: the set of switchable checkpoints is read from
`vendor/registry.json`, snapshotted from `npe_inference/configs/
models.py` by the same `--refresh`. The checkpoints themselves are
data, not code: they stay on the filesystem under `MODEL_WORKDIR`, and
an entry whose checkpoint is not on this machine is dropped rather
than failing startup.
"""

import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# The app's private copies of the non-standard packages, and the
# snapshot of the model registry beside them. Both written by
# vendor.py; both checked in.
VENDOR_DIR = APP_DIR / 'vendor'
REGISTRY_FILE = VENDOR_DIR / 'registry.json'

# Trained runs live at <MODEL_WORKDIR>/<project>/<run_id>/. Same
# variable name npe_infer.paths uses, so one export covers both.
MODEL_WORKDIR = Path(os.environ.get(
    'NPE_MODEL_WORKDIR',
    '/scratch/tvnguyen/projects/sbi_dsph/trained_models/npe'))


def import_path() -> tuple[Path, ...]:
    """The directories to put on `sys.path`, highest priority first.

    Returns:
        `(VENDOR_DIR,)` - the only place the app's non-PyPI imports
        may come from.
    """
    return (VENDOR_DIR,)


def setup() -> None:
    """Make `tsnpe`, `jgnn` and `dsph_analysis` importable. Idempotent.

    Raises:
        SystemExit: If `vendor/` is missing, which means the checkout
            is incomplete - the app has no other source for them.
    """
    if not VENDOR_DIR.is_dir():
        raise SystemExit(
            f'No vendored packages at {VENDOR_DIR}. They are checked '
            'in; a checkout without them is incomplete. Rebuild them '
            'with `python vendor.py --refresh`.')
    for directory in reversed(import_path()):
        entry = str(directory)
        if directory.is_dir() and entry not in sys.path:
            # Reason: position 0, so these win over anything else
            # installed in the environment under the same name -
            # including an editable install of jgnn, whose import hook
            # would otherwise be consulted.
            sys.path.insert(0, entry)


def vendored_versions() -> str:
    """One line naming the commit each vendored package came from.

    Printed at startup and recorded in the run log, so a result can be
    traced to the dependency code that produced it.

    Returns:
        `'name abcd1234, ...'`, or a note if the manifest is missing.
    """
    manifest = VENDOR_DIR / 'VENDOR.json'
    if not manifest.is_file():
        return 'no VENDOR.json (run `python vendor.py --refresh`)'
    packages = json.loads(manifest.read_text()).get('packages', {})
    return ', '.join(
        f'{name} {info.get("commit", "")[:8] or "no-git"}'
        f'{"+dirty" if info.get("dirty") else ""}'
        for name, info in packages.items())


def load_registry() -> dict[str, dict]:
    """The checkpoints this server may switch between, by CLI name.

    Read from `vendor/registry.json`, snapshotted from
    `npe_inference/configs/models.py` - the file where a model's
    radius convention is asserted, and which explains at length why
    the convention is a deliberate per-model declaration and not
    something to be inferred. It is copied verbatim rather than
    restated: two hand-written copies could disagree, and the symptom
    - `dm_log_rdm` off by a factor of r_star - looks like a plausible
    number rather than an error.

    Entries whose checkpoint is not on this filesystem are dropped, so
    a registry naming runs that were never copied here still yields a
    usable subset instead of failing at startup.

    Returns:
        `{name: {name, model_dir, checkpoint, radius_units, note}}`,
        plus `prior_min`/`prior_max` where the model declares them.
        Empty if there is no snapshot (a bundle, which carries its
        models described by their own `model_spec.json`).
    """
    if not REGISTRY_FILE.is_file():
        return {}
    models = json.loads(REGISTRY_FILE.read_text())

    registry = {}
    for spec in models:
        checkpoints = (MODEL_WORKDIR / spec['project'] / spec['run_id']
                       / 'checkpoints')
        if not (checkpoints / spec['checkpoint']).is_file():
            continue
        # prior_min/prior_max travel with the entry: app.py builds the
        # box sample_posterior cuts against, and a model whose training
        # prior is narrower than tsnpe's default (priorC) would
        # otherwise keep draws it was never trained on.
        bounds = {key: spec[key] for key in ('prior_min', 'prior_max')
                  if spec.get(key) is not None}
        registry[spec['name']] = dict(
            name=spec['name'], model_dir=checkpoints,
            checkpoint=spec['checkpoint'],
            radius_units=spec['radius_units'], note=spec['note'],
            **bounds)
    return registry
