"""Refresh the app's private copies of the non-standard packages.

The app imports `tsnpe`, `jgnn` and `dsph_analysis` from `vendor/`,
never from a checkout elsewhere on the machine. That is deliberate:
those three are research code under active development, and an
improvement to one of them used to reach the running app the instant
it was committed, with no record of which version a result came from.
A change to `dsph_analysis`'s Jeans solver did exactly that and cost a
day - the app had not changed at all.

So `vendor/` is checked in, and updating a dependency is an explicit
commit here rather than a side effect of working somewhere else:

    python vendor.py              # what is vendored vs what is live
    python vendor.py --refresh    # re-copy, then review and commit

The sources are only consulted by this script. Nothing the server
imports at runtime knows they exist.

The same copies are what `package.py` ships in a bundle, so a bundle
and a repo-mode server run byte-identical dependency code.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent
VENDOR_DIR = WEBAPP_DIR / 'vendor'
MANIFEST = VENDOR_DIR / 'VENDOR.json'
REGISTRY = VENDOR_DIR / 'registry.json'

# Sources, overridable exactly as in npe_infer/paths.py. Used by
# --refresh only.
PIPELINE_DIR = Path(os.environ.get(
    'SBI_DSPH_PIPELINE_DIR', str(WEBAPP_DIR.parent)))
MY_MODULES_DIR = Path(os.environ.get(
    'MY_MODULES_DIR', '/home/tvnguyen/my_modules'))
NPE_INFERENCE_DIR = Path(os.environ.get(
    'NPE_INFERENCE_DIR', str(PIPELINE_DIR.parent / 'npe_inference')))

# Only what inference needs. tsnpe's training and simulation modules
# and jgnn's callbacks/datasets/training pull in wandb, h5py and tarp,
# none of which a pretrained model has any use for.
TSNPE_MODULES = ('__init__.py', 'target.py', 'prior.py',
                 'model_io.py', 'proposal.py')
JGNN_SUBPACKAGES = ('models', 'transforms')

# Copied whole, and located by import so however they are installed is
# however they are copied. gh_alternative is here because
# dsph_analysis.vkurtosis imports its line profiles - a dependency of
# a dependency is still a dependency the app must not reach outside
# for.
IMPORTED_PACKAGES = ('jgnn', 'dsph_analysis', 'gh_alternative')
JGNN_INIT = '''"""Jeans GNN package (webapp copy: models + transforms only).

Trimmed by webapp/vendor.py from the full jgnn package - training,
callbacks, and dataset modules (and their wandb/h5py dependencies) are
not needed to run a pretrained model.
"""

from . import models
from . import transforms

__all__ = ['models', 'transforms']
'''


def _git(repo: Path, *args: str) -> str:
    """Run a git command in `repo`, or return '' if that is not possible.

    Args:
        repo: Directory inside the working tree to ask about.
        *args: Arguments after `git`.

    Returns:
        Stripped stdout, or '' when the command fails (not a checkout,
        no git, and so on).
    """
    try:
        out = subprocess.run(
            ('git', '-C', str(repo)) + args,
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ''
    return out.stdout.strip() if out.returncode == 0 else ''


def _provenance(repo: Path) -> dict:
    """Where a vendored copy came from, as far as git can say.

    Args:
        repo: Any directory inside the source checkout.

    Returns:
        `{path, commit, dirty}`; `commit` is '' for a non-checkout.
    """
    return dict(
        path=str(repo),
        commit=_git(repo, 'rev-parse', 'HEAD'),
        # Scoped to the copied subtree: unrelated edits elsewhere in
        # the same checkout say nothing about this copy.
        dirty=bool(_git(repo, 'status', '--porcelain', '-uno', '--',
                        str(repo))),
    )


def _copy_tree(src: Path, dst: Path) -> None:
    """Copy a package directory, leaving build droppings behind.

    Args:
        src: Directory to copy.
        dst: Destination, replaced if it exists.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        '__pycache__', '*.pyc', '.git', '.gitignore', '*.ipynb',
        '.ipynb_checkpoints'))


def _source_dirs() -> dict[str, Path]:
    """Locate each source package.

    jgnn and dsph_analysis are found by import, so however they are
    installed is however they are copied; tsnpe comes from the
    checkout this file sits in.

    Returns:
        `{name: package directory}`.

    Raises:
        SystemExit: If a package cannot be imported or found.
    """
    # Reason: importing from the webapp dir would find vendor/ first
    # and copy the old copy onto itself.
    sys.path[:] = [p for p in sys.path
                   if Path(p or '.').resolve() != VENDOR_DIR]
    if MY_MODULES_DIR.is_dir():
        sys.path.insert(0, str(MY_MODULES_DIR))
    dirs = {'tsnpe': PIPELINE_DIR / 'tsnpe' / 'tsnpe'}
    for name in IMPORTED_PACKAGES:
        try:
            module = __import__(name)
        except ImportError as e:
            raise SystemExit(f'Cannot import {name} to copy it: {e}')
        # gh_alternative is a namespace package (no __init__.py), so
        # it has a __path__ but no __file__.
        where = (Path(module.__file__).parent if module.__file__
                 else Path(list(module.__path__)[0]))
        dirs[name] = where.resolve()
    missing = [n for n, d in dirs.items() if not d.is_dir()]
    if missing:
        raise SystemExit(f'Source missing: {missing}')
    return dirs


def _snapshot_registry() -> list[dict]:
    """The model registry as plain data, read from npe_inference.

    `configs/models.py` is where a model's radius convention is
    asserted, and inferring it elsewhere is exactly the mistake
    `tsnpe/prior.py` warns about - so it is snapshotted verbatim here
    rather than restated, and refreshed by the same command that
    refreshes the code.

    Returns:
        One dict per model, in registry order.

    Raises:
        SystemExit: If the registry cannot be imported.
    """
    entry = str(NPE_INFERENCE_DIR)
    if entry not in sys.path:
        sys.path.insert(0, entry)
    try:
        from configs.models import MODELS
    except ImportError as e:
        raise SystemExit(
            f'Cannot import the registry from {NPE_INFERENCE_DIR}: {e}')
    out = []
    for name, spec in MODELS.items():
        item = dict(name=name, project=spec.project, run_id=spec.run_id,
                    checkpoint=spec.checkpoint,
                    radius_units=spec.radius_units, note=spec.note)
        for key in ('prior_min', 'prior_max'):
            if spec.get(key) is not None:
                item[key] = dict(spec[key])
        out.append(item)
    return out


def refresh() -> None:
    """Re-copy every vendored package and rewrite the manifest."""
    sources = _source_dirs()
    VENDOR_DIR.mkdir(exist_ok=True)

    tsnpe_dst = VENDOR_DIR / 'tsnpe'
    if tsnpe_dst.exists():
        shutil.rmtree(tsnpe_dst)
    tsnpe_dst.mkdir()
    for name in TSNPE_MODULES:
        shutil.copy(sources['tsnpe'] / name, tsnpe_dst / name)

    jgnn_dst = VENDOR_DIR / 'jgnn'
    if jgnn_dst.exists():
        shutil.rmtree(jgnn_dst)
    jgnn_dst.mkdir()
    for sub in JGNN_SUBPACKAGES:
        _copy_tree(sources['jgnn'] / sub, jgnn_dst / sub)
    (jgnn_dst / '__init__.py').write_text(JGNN_INIT)

    for name in ('dsph_analysis', 'gh_alternative'):
        _copy_tree(sources[name], VENDOR_DIR / name)

    models = _snapshot_registry()
    REGISTRY.write_text(json.dumps(models, indent=2) + '\n')

    manifest = dict(
        refreshed_at=datetime.now(timezone.utc).isoformat(
            timespec='seconds'),
        packages={name: _provenance(path)
                  for name, path in sources.items()},
        registry=_provenance(NPE_INFERENCE_DIR),
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')

    print(f'Vendored into {VENDOR_DIR}:')
    for name, info in manifest['packages'].items():
        print(f'  {name:14s} {info["commit"][:8] or "no-git":8s}'
              f'{"  (source was dirty)" if info["dirty"] else ""}')
    print(f'  registry.json  {len(models)} models from '
          f'{manifest["registry"]["commit"][:8] or "no-git"}')
    print('\nReview the diff and commit it - that commit is what ties '
          'the app to these versions.')


# Run in a fresh interpreter by --verify. Imports everything the app
# imports that is not on PyPI and insists each one came from vendor/ -
# the editable install of jgnn and any checkout on this machine are
# live adversaries, so this is worth asserting rather than assuming.
_VERIFY = """
import warnings
warnings.filterwarnings('ignore')
import app_paths
app_paths.setup()
import tsnpe, jgnn, dsph_analysis, gh_alternative
from tsnpe import prior, target, model_io, proposal
from jgnn import models, transforms
from dsph_analysis import (sph_model, vdisp, vkurtosis, kinematic_io,
                           data_utils)
from gh_alternative import line_profiles

vendor = str(app_paths.VENDOR_DIR)
mods = [tsnpe, jgnn, dsph_analysis, gh_alternative, prior, target,
        model_io, proposal, models, transforms, sph_model, vdisp,
        vkurtosis, kinematic_io, data_utils, line_profiles]
bad = []
for m in mods:
    # A namespace package (gh_alternative) has __path__, not __file__.
    where = m.__file__ or list(m.__path__)[0]
    if not where.startswith(vendor):
        bad.append((m.__name__, where))
if bad:
    for name, where in bad:
        print('  OUTSIDE vendor/: %s from %s' % (name, where))
    raise SystemExit(1)
print('  %d non-PyPI modules, all from vendor/' % len(mods))
"""


def verify() -> int:
    """Import what the app imports and prove it all came from vendor/.

    Returns:
        0 if every non-PyPI import resolved inside `vendor/`.
    """
    print('Verifying import isolation:', flush=True)
    done = subprocess.run([sys.executable, '-c', _VERIFY],
                          cwd=str(WEBAPP_DIR))
    return done.returncode


def check() -> int:
    """Compare the vendored copies against the live checkouts.

    Purely informational: the app never imports the live copies, so
    drift here is a nudge to run --refresh, not a problem with the
    running server.

    Returns:
        0 if everything matches (or cannot be compared), 1 if any
        source has moved on.
    """
    if not MANIFEST.is_file():
        print(f'No {MANIFEST.name}: run `python vendor.py --refresh`.')
        return 1
    manifest = json.loads(MANIFEST.read_text())
    print(f'Vendored {manifest["refreshed_at"]}')
    behind = []
    recorded = dict(manifest['packages'])
    recorded['registry'] = manifest['registry']
    for name, info in recorded.items():
        live = _git(Path(info['path']), 'rev-parse', 'HEAD')
        if not info['commit'] or not live:
            state = 'not a checkout - cannot compare'
        elif live == info['commit']:
            state = 'up to date'
        else:
            state = f'live checkout is at {live[:8]}'
            behind.append(name)
        print(f'  {name:14s} {info["commit"][:8] or "no-git":8s} {state}')
    if behind:
        print('\nBehind: ' + ', '.join(behind)
              + '\nThe app keeps running the vendored code until you '
                'run `python vendor.py --refresh` and commit.')
    return 1 if behind else 0


def main() -> int:
    """Parse arguments and run the requested action.

    Returns:
        Process exit status.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--refresh', action='store_true',
        help='re-copy from the live checkouts and rewrite VENDOR.json')
    parser.add_argument(
        '--verify', action='store_true',
        help='check that every non-PyPI import resolves inside vendor/')
    args = parser.parse_args()
    if args.refresh:
        refresh()
    if args.verify:
        return verify()
    return 0 if args.refresh else check()


if __name__ == '__main__':
    sys.exit(main())
