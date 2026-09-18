"""Round >= 1 simulation: the training set's own physics, via dsph_sims.

`simulate_round.py` hands this module the proposal in model units
(labels + conditioning, the run Prior's `all_names` order). Each galaxy
is mapped to the spec's sampling space with `TsnpePrior.model_to_sim`,
then simulated and cut by `dsph_sims.runner.simulate_one` -- the very
function that built the round-0 dataset -- and written back in model
units, which is what train_round.py reads. Output is raw Cartesian
pos/vel; sky-plane projection happens at train time as a pre_transform
(see tsnpe/proposal.py's docstring).
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm

from dsph_sims import get_spec
from dsph_sims.models import common
from dsph_sims.runner import simulate_one as _simulate_sampling_space


def simulate_one(
    theta_model: np.ndarray, num_stars: int, spec_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Simulate and preprocess a single galaxy given in model units.

    Runs in a worker process; only the spec's name crosses the process
    boundary.

    Args:
        theta_model: Length len(all_names) row, model units.
        num_stars: Number of stars to sample for this galaxy.
        spec_name: dsph_sims spec of the run's training set.

    Returns:
        (theta_model, posvel) if accepted, or (None, None) if rejected at
        either simulation or preprocessing.
    """
    spec = get_spec(spec_name)
    theta_sim = spec.tsnpe.model_to_sim(
        np.asarray(theta_model, dtype=float)[None])[0]
    _, posvel, _ = _simulate_sampling_space(spec_name, theta_sim, num_stars)
    if posvel is None:
        return None, None
    return theta_model, posvel


def run_simulation_batch(
    params: np.ndarray,
    num_stars: list[int],
    spec_name: str,
    n_jobs: int = 0,
    use_multiprocessing: bool = True,
    sample_threads: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate a batch of galaxies, keeping only the successful draws.

    Args:
        params: (N, len(all_names)) rows in model units, one per galaxy.
        num_stars: Length-N sequence of star counts to draw per galaxy.
        spec_name: dsph_sims spec of the run's training set.
        n_jobs: Worker processes to use (0 -> the CPUs this process may
            actually run on, i.e. the SLURM allocation, not the whole
            node). Ignored if `use_multiprocessing` is False.
        use_multiprocessing: If False, simulate serially in this process
            (useful for debugging).
        sample_threads: OpenMP threads each worker process may use
            internally for agama calls.

    Returns:
        Tuple of (theta, posvel_list): `theta` is the (n_success, D) subset
        of `params` that were accepted, and `posvel_list` is the matching
        list of (n_i, 6) position/velocity arrays.

    Raises:
        KeyError: If `spec_name` is not registered.
        ValueError: If the spec has no tsnpe adapter.
    """
    spec = get_spec(spec_name)
    if spec.tsnpe is None:
        raise ValueError(f'model spec {spec_name!r} has no tsnpe adapter')
    params = np.asarray(params)
    theta_list, posvel_list = [], []

    if not use_multiprocessing:
        for p, n in tqdm(
            zip(params, num_stars), total=len(params), desc='Simulating',
        ):
            p_out, pv = simulate_one(p, int(n), spec_name)
            if pv is not None:
                theta_list.append(p_out)
                posvel_list.append(pv)
        return np.array(theta_list), posvel_list

    n_workers = n_jobs or len(os.sched_getaffinity(0))
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=common.init_worker,
        initargs=(sample_threads,),
    ) as pool:
        futures = [
            pool.submit(simulate_one, p, int(n), spec_name)
            for p, n in zip(params, num_stars)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc='Simulating'):
            p_out, pv = fut.result()
            if pv is not None:
                theta_list.append(p_out)
                posvel_list.append(pv)

    return np.array(theta_list), posvel_list


def write_graph_dataset(
    path: str,
    theta: np.ndarray,
    posvel_list: list[np.ndarray],
    param_names: list[str],
    headers: dict | None = None,
) -> None:
    """Write a batch of simulated galaxies to an HDF5 graph dataset.

    Layout matches what `jgnn.datasets.io.read_graph_dataset` /
    `jgnn.datasets.cartesian` expect: node features 'pos'/'vel' (each
    galaxy's stars concatenated together, split back out via 'ptr'), and
    one graph-level dataset per entry in `param_names`.

    Args:
        path: Destination HDF5 path.
        theta: (n_galaxies, len(param_names)) model-unit parameters.
        posvel_list: Length-n_galaxies list of (n_i, 6) pos/vel arrays.
        param_names: Names for `theta`'s columns, in order.
        headers: Extra scalar attributes to store (e.g. round, tau).
    """
    theta = np.asarray(theta)
    num_stars = [pv.shape[0] for pv in posvel_list]
    ptr = np.cumsum([0] + num_stars)
    pos = np.concatenate([pv[:, :3] for pv in posvel_list], axis=0)
    vel = np.concatenate([pv[:, 3:6] for pv in posvel_list], axis=0)

    graph_features = list(param_names) + ['num_stars', 'ptr']
    all_features = ['pos', 'vel'] + graph_features

    with h5py.File(path, 'w') as f:
        f.create_dataset('pos', data=pos)
        f.create_dataset('vel', data=vel)
        f.create_dataset('num_stars', data=num_stars)
        f.create_dataset('ptr', data=ptr)
        for i, name in enumerate(param_names):
            f.create_dataset(name, data=theta[:, i])

        f.attrs['all_features'] = all_features
        f.attrs['node_features'] = ['pos', 'vel']
        f.attrs['graph_features'] = graph_features
        for key, value in (headers or {}).items():
            f.attrs[key] = value
