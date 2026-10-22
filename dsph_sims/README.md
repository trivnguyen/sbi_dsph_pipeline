# dsph_sims

Simulators for the dSph kinematics training sets, as one importable
package (editable-installed from the pipeline root: `pip install -e .`).
The unit of organisation is a `ModelSpec` (`spec.py`): the sampling-space
parameters and their uniform box, the agama physics, the acceptance cuts,
the columns stored per galaxy, the default NPE labels/conditioning, the
datasets it produced, and — where tsnpe can use it — a `TsnpePrior`
adapter (box in r_star units, kpc-offset columns, model -> simulator map).

```
dsph_sims/
  __init__.py       public API: SPECS, get_spec, spec_for_dataset,
                    dataset_model_spec
  spec.py           ModelSpec and TsnpePrior
  runner.py         batch simulation -> data.<i>.h5 + config.<i>.json
  models/           one spec per file; models/common.py holds the shared
                    agama setup, acceptance cuts and half-light radius
  scripts/          simulate_batch.py (the CLI)
  legacy/           the original simulate_*.py scripts, frozen verbatim
  tests/            test_specs.py (bit-exact goldens + dataset checks)
  notebooks/        prior and light-profile / flattened-halo studies
```

```python
from dsph_sims import get_spec, spec_for_dataset, dataset_model_spec
spec = get_spec('8p_ZhaoPlumCOM_v4')
spec_for_dataset('8p_ZhaoPlumCOM_v4') is spec          # legacy lookup by name
dataset_model_spec(data_root, '8p_ZhaoPlumCOM_v4')     # from config.0.json
```

| spec                     | module            | datasets                       | tsnpe |
|--------------------------|-------------------|--------------------------------|-------|
| `8p_ZhaoPlumCOM` | `models/ZhaoPlumCOM_8p.py` | `8p_ZhaoPlumCOM`, `_v2`        | yes   |
| `8p_ZhaoPlumCOM_v3` | `models/ZhaoPlumCOM_8p_v3.py` | `8p_ZhaoPlumCOM_v3`            | yes   |
| `8p_ZhaoPlumCOM_v4` | `models/ZhaoPlumCOM_8p_v4.py` | `8p_ZhaoPlumCOM_v4`            | yes   |
| `9p_ZhaoPlumGammaCOM`           | `ZhaoPlumGammaCOM_9p.py`    | `9p_ZhaoPlumGammaCOM`                 | no    |
| `10p_FlatZhaoPlumCOM`           | `FlatZhaoPlumCOM_10p.py`     | `10p_FlatZhaoPlumCOM`              | no    |
| `11p_ZhaoABGCOM`                | `ZhaoABGCOM_11p.py`          | `11p_ZhaoABGCOM`                      | no    |
| `10p_ZhaoPlumJeansBVH_twin` / `_ktwin` | `models/ZhaoPlumJeansBVH_10p_v1.py` | sbi_twins `datasets/bvh10p_500k/*` | no |
| `14p_ZhaoPlumJeansBVH_twin` / `_ktwin` | `models/ZhaoPlumJeansBVH_14p_v1.py` | sbi_twins `datasets/bvh14p_500k/*` | no |
| `14p_ZhaoPlumJeansBVH_v2_twin` / `_ktwin` | `models/ZhaoPlumJeansBVH_14p_v2.py` | none yet; v1 + prior constraint `df_betasyminf >= df_betasym0` | no |

The 14p Jeans specs are the odd ones out: DF-free (no agama), positions
from the Plummer profile and line-of-sight-only velocities from the Jeans
moments, NumPy randomness with one seed per galaxy (`seeded=True`),
rejection *reasons* counted into `config.<i>.json`, and per-galaxy
diagnostics stored as extra columns. The sbi_twins helpers are inlined
(the Sanders & Evans kernel as `models/sanders_evans.py`); only the current
`dsph_analysis` is needed, at simulation time. See the module docstring.

## Running

```bash
python scripts/simulate_batch.py --list
python scripts/simulate_batch.py --model 8p_ZhaoPlumCOM_v4 \
    --n-sims 100000 --output-dir /scratch/.../8p_ZhaoPlumCOM_v5 --n-workers 192
# or on SLURM, extra args are forwarded:
sbatch ../npe/slurm/simulate.sbatch ../dsph_sims/scripts/simulate_batch.py \
    /scratch/.../8p_ZhaoPlumCOM_v5 100000 --model 8p_ZhaoPlumCOM_v4
```

`runner.py` is the ProcessPool + HDF5 batch runner the per-family scripts
used to carry. It draws exactly as they did (uniform box, constraint
rejection, Poisson star count — same numpy consumption, so a dataset is
reproducible from the seed in its config), writes `data.<i>.h5` in the
layout `jgnn.datasets.io.read_graph_dataset` expects, and records the
spec name, box and constraint in `config.<i>.json` and as an HDF5 attr.

## Adding a model

1. Write `models/<family>.py` (one file per prior variant, explicit
   rather than parametrized - see the three `ZhaoPlumCOM_8p*.py`): `PARAM_NAMES`, `simulate(num_stars, params) ->
   (posvel, extras) | None`, `preprocess(posvel, params)` (usually
   `common.preprocess` with the family's `r_star`), `columns(params,
   extras)`, and a `ModelSpec`. Give it a `TsnpePrior` only if tsnpe can
   condition on it (one axis, log10 r_half/kpc).
2. Register it in `models/__init__.py`.
3. Simulate; put `config.model_spec` in the npe config and the tsnpe
   config. Nothing else needs to know about it.
4. Add goldens to `tests/test_specs.py` once the physics is final.

## Provenance

`legacy/` holds the original scripts verbatim — the code that actually
produced every dataset listed above (`simulate_8params_thread.py` there
is the older ThreadPoolExecutor variant, unused). `tests/test_specs.py`
carries sha256
goldens generated from them (posvel from a fixed agama seed, prior draws
from a fixed numpy seed) and checks each dataset directory's
`config.*.json` against its spec; run it after touching any physics:

```bash
python tests/test_specs.py
```
