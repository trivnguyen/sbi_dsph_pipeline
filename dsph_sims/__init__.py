"""Registry of simulator model specs; see `spec.ModelSpec`.

    from dsph_sims import get_spec, spec_for_dataset
    spec = get_spec('8p_ZhaoPlumCOM_v4')
    spec = spec_for_dataset('8p_ZhaoPlumCOM_v4')   # same object

Layout: `spec.py` (the dataclasses), `models/` (one spec per file, and
`models/common.py` with the shared agama physics), `runner.py` (batch
simulation to HDF5), `scripts/simulate_batch.py` (CLI), `legacy/` (the
original scripts, frozen), `tests/`, `notebooks/`. Importing this package
imports agama (the families need it).
"""

import json
import os

from .models import SPECS
from .spec import ModelSpec, TsnpePrior

_DATASET_TO_SPEC = {
    d: s.name for s in SPECS.values() for d in s.datasets
}
assert len(_DATASET_TO_SPEC) == sum(len(s.datasets) for s in SPECS.values()), \
    'a dataset name is claimed by two specs'


def get_spec(name: str) -> ModelSpec:
    """Look a spec up by name.

    Raises:
        KeyError: Listing the registered names.
    """
    try:
        return SPECS[name]
    except KeyError:
        raise KeyError(
            f'unknown model spec {name!r}; registered: {sorted(SPECS)}'
        ) from None


def spec_for_dataset(data_name: str) -> ModelSpec:
    """The spec that produced a dataset directory, by its name.

    Raises:
        KeyError: If no spec claims `data_name`.
    """
    try:
        return SPECS[_DATASET_TO_SPEC[data_name]]
    except KeyError:
        raise KeyError(
            f'no model spec claims dataset {data_name!r}; known: '
            f'{sorted(_DATASET_TO_SPEC)}') from None


def dataset_model_spec(data_root, data_name: str) -> str:
    """The spec name a dataset directory was simulated with.

    Read from the directory's own config.0.json (written by the runner,
    and back-filled into every existing dataset), falling back to the
    registry's `datasets` lists for a directory that is gone or was made
    before the key existed.

    Args:
        data_root: Directory holding the dataset directories, or None.
        data_name: Dataset directory name.

    Returns:
        A key of `SPECS`.

    Raises:
        KeyError: If neither source knows the dataset.
    """
    if data_root:
        path = os.path.join(data_root, data_name, 'config.0.json')
        if os.path.exists(path):
            name = json.load(open(path)).get('model_spec')
            if name:
                return get_spec(name).name
    return spec_for_dataset(data_name).name


__all__ = ['ModelSpec', 'TsnpePrior', 'SPECS', 'get_spec',
           'spec_for_dataset', 'dataset_model_spec']
