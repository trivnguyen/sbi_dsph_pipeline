"""Build a self-contained, distributable copy of the web app.

Assembles everything the app needs into one directory - the app itself,
the trained model, and private copies of the `tsnpe`, `jgnn` (models +
transforms only), and `dsph_analysis` packages - so the result can be
copied to any machine, pip-installed from its requirements.txt (or
built with its Dockerfile), and launched with no access to this repo,
the cluster filesystem, or the network:

    python package.py --checkpoint-dir /path/to/checkpoints
    cd dist/dsph_explorer
    pip install -r requirements.txt
    python app.py

Usage:
    python package.py --checkpoint-dir DIR [--checkpoint-filename F]
                      [--out DIR]
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

_WEBAPP_DIR = Path(__file__).resolve().parent
_REPO_TSNPE = _WEBAPP_DIR.parent / 'tsnpe' / 'tsnpe'

_APP_FILES = ('app.py', 'inference.py', 'user_catalog.py')
_TSNPE_MODULES = ('__init__.py', 'target.py', 'prior.py',
                  'model_io.py', 'proposal.py')

# jgnn's real __init__ also imports callbacks/datasets/training/utils,
# which drag in wandb/h5py/tarp - none of it needed at inference time.
_JGNN_INIT = '''"""Jeans GNN package (webapp bundle: models + transforms only).

Trimmed by webapp/package.py from the full jgnn package - training,
callbacks, and dataset modules (and their wandb/h5py dependencies) are
not needed to run the pretrained model.
"""

from . import models
from . import transforms

__all__ = ['models', 'transforms']
'''

_REQUIREMENTS = '''\
# Core inference stack. torch/torch-cluster often need a
# platform-specific install first - see README.md.
torch>=2.4
# Pinned: newer torch-geometric releases changed knn_graph's fallback
# and require pyg-lib - stick to the version this model was actually
# trained/validated against.
torch-geometric==2.7.0
torch-cluster>=1.6.3
pytorch-lightning>=2.4
zuko>=1.3
ml-collections>=1.0
numpy>=1.26
scipy>=1.13
pandas>=2.2
astropy>=6.0
emcee>=3.1
corner>=2.2
matplotlib>=3.8
tqdm>=4.66
# Web server.
fastapi>=0.110
uvicorn>=0.29
python-multipart>=0.0.9
'''

_DOCKERFILE = '''\
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch, then torch-cluster from the matching PyG wheel index
# (adjust the torch version in the URL if you change the pin), then the
# rest. For GPU serving, swap in the CUDA wheel indices instead.
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.* \\
        --index-url https://download.pytorch.org/whl/cpu \\
    && pip install --no-cache-dir torch-cluster \\
        -f https://data.pyg.org/whl/torch-2.9.0+cpu.html \\
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8799
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8799"]
'''

_BUNDLE_README = '''\
# dSph posterior explorer (self-contained bundle)

Amortized dwarf-spheroidal density-profile inference in the browser:
upload a kinematic catalog, fill in the system metadata, and get an
interactive profile explorer (density, enclosed mass, anisotropy, LOS
dispersion, LOS kurtosis), the posterior corner plot, a Wolf-mass
sanity check, and the raw posterior samples as CSV.

Everything is local to this directory: the app, the trained model
(`model/`), and private copies of the analysis packages it needs. No
repository access or network access is required at runtime.

## Run

```bash
pip install -r requirements.txt   # see note below for torch
python app.py                     # http://localhost:8799
```

`torch` and `torch-cluster` sometimes need a platform-specific install
before `pip install -r requirements.txt` - e.g. CPU-only:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.9.0+cpu.html
```

Or build the Docker image (CPU-only by default):

```bash
docker build -t dsph-explorer .
docker run -p 8799:8799 dsph-explorer
```

A GPU is optional - on CPU a typical run (1000 posterior samples, 500
profile samples) takes a few minutes instead of ~20 s.

Options: `python app.py --help` (port, device, model directory,
profile-worker count).

There is no authentication - only expose the port to people you would
let run jobs on the host machine.

## Data format

One row per star; CSV, ECSV, or FITS table. Required columns
(case-insensitive, common alias spellings accepted, and every
assignment can be corrected by hand in the UI after upload):

| column     | unit | meaning                                    |
|------------|------|--------------------------------------------|
| `ra`       | deg  | right ascension (ICRS)                     |
| `dec`      | deg  | declination (ICRS)                         |
| `vr`       | km/s | heliocentric line-of-sight velocity        |
| `vr_err`   | km/s | its 1-sigma uncertainty                    |
| `distance` | kpc  | per-star distance, **or** `dm` [mag]; the
                     median sets the system distance (or type the
                     distance into the form instead). |

Optional, auto-detected: a membership-probability column (threshold
cut in the UI) and arbitrarily-named boolean flag columns (each can be
ignored, required true, or required false).

`example_catalog.csv` is a small synthetic file in the right format -
try it with R_half = 0.2 kpc.

## Selecting rows

Two more filters sit under the upload box, and combine with everything
above (all cuts are ANDed):

- **Row filter** - a [pandas query](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html)
  expression over your file's columns, e.g. `key == "draco_1"`,
  `mem_prob > 0.8 and good_star`, `key in ["draco_1", "bootes_1"]`.
  One file holding several systems is the main use: the filter picks
  the one to fit, and the center, distance, and systemic velocity are
  then measured from just those stars. In compare mode dataset B can
  reuse A's file with a different filter, so A vs B can be two systems
  out of one catalog.
- **Radius cut** - min/max on the projected radius from the center, in
  kpc or arcmin.

The projected radius is also available to the row filter as `R_kpc` and
`R_arcmin` (e.g. `R_kpc < 5`). These are computed from the center, so
they replace same-named columns in your file.

## System metadata

The half-light radius **R_half [kpc]** is required - the model
conditions on it. Center, systemic velocity, and distance default to
data-driven medians when left blank; proper motions are only needed
for the perspective-rotation correction. Systems in the bundled
local_volume_database snapshot can be prefilled by key.

## Swapping in another model

Replace the contents of `model/` with any checkpoint trained by this
pipeline: the Lightning `.ckpt` plus the `config_snapshot.json`
written next to it at training time. If the directory holds exactly
one `.ckpt`, any file name works.

## Deploying on a website

This is a Python web service, so it needs a host that can run a
process (a VPS, lab server, or container platform) - it cannot run on
static-only hosting (GitHub Pages, plain shared hosting). A 1-2 CPU /
2 GB RAM box is enough.

Typical setup - run the app as a service and put your web server in
front of it:

```ini
# /etc/systemd/system/dsph-explorer.service
[Unit]
Description=dSph posterior explorer
After=network.target

[Service]
WorkingDirectory=/opt/dsph_explorer
ExecStart=/opt/dsph_explorer/.venv/bin/python app.py \\
    --host 127.0.0.1 --port 8799 --output-dir /opt/dsph_explorer/runs
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```nginx
# nginx: serve it under your domain (add HTTPS with certbot)
location /dsph/ {
    proxy_pass http://127.0.0.1:8799/;
    proxy_read_timeout 600s;   # runs can take minutes on CPU
    client_max_body_size 50m;  # catalog uploads
    # optional gate, since the app has no auth of its own:
    # auth_basic "dsph explorer";
    # auth_basic_user_file /etc/nginx/.htpasswd;
}
```

With `--host 127.0.0.1` the app is only reachable through the proxy.
Anyone who can reach the page can submit runs on your machine, so add
the basic-auth lines (or equivalent) if the URL is public.
'''


def _example_catalog(path: Path) -> None:
    """Write a small synthetic catalog in the documented format."""
    rng = np.random.default_rng(42)
    n = 250
    ra = 260.05 + rng.normal(0, 0.08, n)
    dec = 57.9 + rng.normal(0, 0.08, n)
    dist = rng.normal(80.0, 0.5, n)
    vr = -290.0 + rng.normal(0, 9.0, n)
    vr_err = np.abs(rng.normal(2.0, 0.5, n)) + 0.5
    mem_prob = np.clip(rng.uniform(0.5, 1.0, n), 0, 1)
    good = rng.random(n) > 0.05
    header = 'ra,dec,distance,vr,vr_err,mem_prob,good_star\n'
    rows = ''.join(
        f'{ra[i]:.6f},{dec[i]:.6f},{dist[i]:.2f},{vr[i]:.3f},'
        f'{vr_err[i]:.3f},{mem_prob[i]:.3f},{str(good[i])}\n'
        for i in range(n))
    path.write_text(header + rows)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(
        src, dst, ignore=shutil.ignore_patterns(
            '__pycache__', '*.pyc', '.git', '.git*', '*.ipynb'))


def build(checkpoint_dir: Path, checkpoint_filename: str,
          out: Path) -> None:
    """Assemble the bundle at `out` (replacing any previous build)."""
    sys.path.insert(0, str(_WEBAPP_DIR))
    import inference  # noqa: E402 - needs webapp dir on sys.path

    checkpoint_path = checkpoint_dir / checkpoint_filename
    if not checkpoint_path.exists():
        raise FileNotFoundError(f'No checkpoint at {checkpoint_path}')
    snapshot = inference._find_config_snapshot(checkpoint_path)
    if snapshot is None:
        raise FileNotFoundError(
            f'No config_snapshot.json found near {checkpoint_path}')

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # The app itself.
    for name in _APP_FILES:
        shutil.copy(_WEBAPP_DIR / name, out / name)
    _copy_tree(_WEBAPP_DIR / 'static', out / 'static')

    # The model.
    model_dir = out / 'model'
    model_dir.mkdir()
    shutil.copy(checkpoint_path, model_dir / 'model.ckpt')
    shutil.copy(snapshot, model_dir / 'config_snapshot.json')

    # Vendored packages. jgnn/dsph_analysis are located via import so
    # the script works regardless of how they're installed.
    tsnpe_dst = out / 'tsnpe'
    tsnpe_dst.mkdir()
    for name in _TSNPE_MODULES:
        shutil.copy(_REPO_TSNPE / name, tsnpe_dst / name)

    import dsph_analysis
    import jgnn
    jgnn_src = Path(jgnn.__file__).parent
    jgnn_dst = out / 'jgnn'
    jgnn_dst.mkdir()
    _copy_tree(jgnn_src / 'models', jgnn_dst / 'models')
    _copy_tree(jgnn_src / 'transforms', jgnn_dst / 'transforms')
    (jgnn_dst / '__init__.py').write_text(_JGNN_INIT)
    _copy_tree(Path(dsph_analysis.__file__).parent,
               out / 'dsph_analysis')

    # Support files.
    (out / 'requirements.txt').write_text(_REQUIREMENTS)
    (out / 'Dockerfile').write_text(_DOCKERFILE)
    (out / 'README.md').write_text(_BUNDLE_README)
    _example_catalog(out / 'example_catalog.csv')

    size_mb = sum(
        f.stat().st_size for f in out.rglob('*') if f.is_file()
    ) / 1e6
    print(f'Bundle built at {out} ({size_mb:.0f} MB)')

    # Also (re)build the distributable tarball next to the bundle dir,
    # so `dist/<name>.tar.gz` is never stale relative to the source.
    archive = shutil.make_archive(
        base_name=str(out), format='gztar',
        root_dir=str(out.parent), base_dir=out.name)
    tar_mb = Path(archive).stat().st_size / 1e6
    print(f'Tarball written to {archive} ({tar_mb:.0f} MB)')
    print('Try it:  cd', out, '&& python app.py')


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--checkpoint-dir', required=True,
        help='Directory holding the trained checkpoint, with '
             'config_snapshot.json alongside or a few parents up.')
    parser.add_argument('--checkpoint-filename', default='last.ckpt')
    parser.add_argument(
        '--out', default=str(_WEBAPP_DIR / 'dist' / 'dsph_explorer'),
        help='Output bundle directory (default: webapp/dist/'
             'dsph_explorer).')
    args = parser.parse_args()
    build(Path(args.checkpoint_dir), args.checkpoint_filename,
          Path(args.out))


if __name__ == '__main__':
    main()
