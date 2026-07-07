""" Simulate the 6D stellar kinematics of dwarf galaxies (process pool). """

from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
from glob import glob

# Belt-and-suspenders alongside agama.setNumThreads() below: this
# makes single-threaded OpenMP the process-wide default read at
# each worker's first parallel region, regardless of how that
# worker process's OpenMP runtime gets (re-)initialized after fork.
# Must be set before `import agama`.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import time
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import agama
import astropy.units as u
import h5py
import numpy as np
from tqdm import tqdm

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)

# Each worker here is a separate OS process with its own isolated
# agama state (including its own copy of agama's internal RNG), so
# unlike the ThreadPoolExecutor version, no lock is needed around
# sampling - there's no shared mutable state to race on. Default to
# 1 OpenMP thread per process; _init_worker() below can raise this
# per --sample-threads. Keep n_workers x sample_threads within your
# allocated core count to avoid oversubscription.
agama.setNumThreads(1)

# Parameter space: (alpha, beta, gamma, log_rdm, log_rhos, beta0,
#                    log_ra, log_rstar)
# NOTE: rdm is in kpc, rstar is in unit of rdm, ra is in unit of rstar
PRIOR_MIN = np.array(
    [0.5, 1.0, -1.0, -2.0, 3.0, -0.499, -1.0, -3.0])
PRIOR_MAX = np.array(
    [3.0, 10.0, 2.0, 2.0, 10.0, 1.0, 3.0, 0.0])


def _init_worker(sample_threads: int) -> None:
    """
    Pool initializer: run once in each worker process at startup.

    Args:
        sample_threads: Number of OpenMP threads this worker
            process may use internally for agama calls.
    """
    agama.setNumThreads(sample_threads)


def simulator(
    num_stars: int, params: np.ndarray
) -> np.ndarray | None:
    """
    Sample 6D positions and velocities for stars in a dwarf galaxy.

    Args:
        num_stars: Number of stars to draw from the galaxy model.
        params: Array of 8 model parameters (alpha, beta, gamma,
            log_rdm, log_rhos, beta0, log_ra, log_rstar).

    Returns:
        Array of shape (num_stars, 6) with columns (x, y, z, vx, vy,
        vz), or None if the model could not be constructed.
    """
    alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar = (
        params
    )
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar * r_dm
    r_a = 10 ** log_ra * r_star
    rho_s = 10 ** log_rhos
    try:
        dm_potential = agama.Potential(
            type='Spheroid', alpha=alpha, beta=beta, gamma=gamma,
            scaleRadius=r_dm, densityNorm=rho_s,
            outercutoffRadius=max(50, 10 * r_dm))
        stellar_density = agama.Density(
            type='Plummer', mass=1, scaleRadius=r_star)
        dist_function = agama.DistributionFunction(
            type='QuasiSpherical', potential=dm_potential,
            density=stellar_density, beta0=beta0, r_a=r_a)
        galaxy_model = agama.GalaxyModel(dm_potential, dist_function)
        posvel, _ = galaxy_model.sample(num_stars)
    except Exception:
        return None
    return posvel


def preprocess(
    posvel: np.ndarray, params: np.ndarray
) -> np.ndarray | None:
    """
    Preprocess the simulated data for training.

    Args:
        posvel: Array of shape (num_stars, 6) with positions and
            velocities, as returned by `simulator`.
        params: Array of 8 model parameters (alpha, beta, gamma,
            log_rdm, log_rhos, beta0, log_ra, log_rstar).

    Returns:
        Filtered array of positions and velocities, or None if the
        galaxy does not meet the acceptance criteria.
    """
    # parse the params and data
    _, _, _, log_rdm, _, _, _, log_rstar = params
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar * r_dm

    num_stars = posvel.shape[0]
    rad3d = np.linalg.norm(posvel[:, :3], axis=1)  # 3D radius
    vel3d = np.linalg.norm(posvel[:, 3:6], axis=1)  # 3D velocity
    veldisp3d = np.std(vel3d)  # 3D velocity dispersion

    # default preprocessing settings
    min_v, max_v = 0., 1000.
    min_vdisp, max_vdisp = 1e-10, 1e10
    min_radius_rstar, max_radius_rstar = 0., 10
    min_radius_kpc, max_radius_kpc = 0., 100
    min_radius = max(min_radius_rstar * r_star, min_radius_kpc)
    max_radius = min(max_radius_rstar * r_star, max_radius_kpc)

    if (veldisp3d < min_vdisp) or (veldisp3d > max_vdisp):
        # if the velocity dispersion is outside the range, skip
        return None

    mask1 = (vel3d > min_v) & (vel3d < max_v)
    mask2 = (rad3d > min_radius) & (rad3d < max_radius)
    mask = mask1 & mask2

    if np.sum(mask) < int(num_stars * 0.5):
        # if half of the stars are outside the range, skip
        return None
    posvel = posvel[mask]
    return posvel


def simulate_one(
    params: np.ndarray, num_stars: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Simulate and preprocess a single galaxy.

    Runs in a worker process. Defined at module level so it can be
    pickled and sent to worker processes by `ProcessPoolExecutor`.
    No locking needed - each process has isolated agama state.

    Args:
        params: Array of 8 model parameters.
        num_stars: Number of stars to sample for this galaxy.

    Returns:
        Tuple of (params, posvel) if the galaxy was accepted, or
        (None, None) if it was rejected at either simulation or
        preprocessing.
    """
    pv = simulator(num_stars, params)
    if pv is None:
        return None, None
    pv = preprocess(pv, params)
    if pv is None:
        return None, None
    return params, pv


def save_simdata(
    theta_list: list[np.ndarray],
    posvel_list: list[np.ndarray],
    file_path: str,
) -> None:
    """
    Save a batch of simulated galaxies to an HDF5 file.

    Args:
        theta_list: List of parameter arrays, one per galaxy.
        posvel_list: List of position/velocity arrays, one per
            galaxy.
        file_path: Destination path for the HDF5 file.
    """
    (alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra,
     log_rstar) = np.array(theta_list).T

    # unit conversion to kpc
    log_rstar_kpc = log_rstar + log_rdm
    log_ra_kpc = log_ra + log_rstar_kpc

    # store pos and vel in a single array, and store the number of
    # stars in each galaxy
    num_stars = [pv.shape[0] for pv in posvel_list]
    ptr = np.cumsum([0] + num_stars)
    pos = np.concatenate([pv[:, :3] for pv in posvel_list], axis=0)
    vel = np.concatenate([pv[:, 3:6] for pv in posvel_list], axis=0)

    with h5py.File(file_path, 'w') as f:
        f.create_dataset('dm_alpha', data=alpha)
        f.create_dataset('dm_beta', data=beta)
        f.create_dataset('dm_gamma', data=gamma)
        f.create_dataset('dm_log_rdm', data=log_rdm)
        f.create_dataset('dm_log_rho0', data=log_rhos)
        f.create_dataset('df_beta0', data=beta0)
        f.create_dataset('df_log_ra', data=log_ra)
        f.create_dataset('stellar_log_ra_kpc', data=log_ra_kpc)
        f.create_dataset('stellar_log_rstar', data=log_rstar)
        f.create_dataset('stellar_log_rstar_kpc', data=log_rstar_kpc)
        f.create_dataset('num_stars', data=num_stars)
        f.create_dataset('ptr', data=ptr)
        f.create_dataset('pos', data=pos)
        f.create_dataset('vel', data=vel)

        # headers
        f.attrs['all_features'] = [
            'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
            'dm_log_rho0', 'df_beta0', 'df_log_ra',
            'stellar_log_ra_kpc', 'stellar_log_rstar',
            'stellar_log_rstar_kpc', 'num_stars', 'ptr', 'pos', 'vel'
        ]
        f.attrs['node_features'] = ['pos', 'vel']
        f.attrs['graph_features'] = [
            'dm_alpha', 'dm_beta', 'dm_gamma', 'dm_log_rdm',
            'dm_log_rho0', 'df_beta0', 'df_log_ra',
            'stellar_log_ra_kpc', 'stellar_log_rstar',
            'stellar_log_rstar_kpc', 'num_stars', 'ptr'
        ]


def save_config(args: argparse.Namespace, file_path: str) -> None:
    """
    Save the run configuration for reproducibility.

    Args:
        args: Parsed command line arguments used for this run,
            including the resolved random seed.
        file_path: Destination path for the config JSON file.
    """
    config = vars(args).copy()
    config['prior_min'] = PRIOR_MIN.tolist()
    config['prior_max'] = PRIOR_MAX.tolist()
    config['timestamp'] = datetime.datetime.now().isoformat()
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=2)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Namespace with the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Simulate the 6D stellar kinematics of dwarf '
                     'galaxies (ProcessPoolExecutor version).')
    parser.add_argument(
        '--n-sims', type=int, default=10000,
        help='Number of galaxies to attempt to simulate.')
    parser.add_argument(
        '--n-workers', type=int, default=os.cpu_count(),
        help='Number of worker processes to use.')
    parser.add_argument(
        '--n-stars', type=int, default=100,
        help='Mean number of stars sampled per galaxy.')
    parser.add_argument(
        '--sample-threads', type=int, default=1,
        help='Number of OpenMP threads each worker process may use '
             'internally for agama calls. Keep n-workers x '
             'sample-threads within your allocated core count to '
             'avoid oversubscription. Tune this per-cluster.')
    parser.add_argument(
        '--galaxies-per-file', type=int, default=1000,
        help='Maximum number of successful galaxies stored in each '
             'output HDF5 file.')
    parser.add_argument(
        '--output-dir', type=str, default='./simdata',
        help='Directory to create (if needed) and save output '
             'files to.')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility.')
    parser.add_argument(
        '--max-pending', type=int, default=None,
        help='Maximum number of in-flight simulations kept in '
             'memory at once (default: 4x n-workers). Lower this '
             'if simulations run out of memory.')
    parser.add_argument(
        '--append', action='store_true',
        help='Append to existing output files instead of overwriting '
             'them. If set, the output directory must already exist.')
    return parser.parse_args()


def iter_params(
    n_sims: int, n_stars: int, rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, int]]:
    """
    Lazily draw (params, num_stars) pairs from the prior.

    Draws are generated one at a time instead of allocating arrays
    of length `n_sims` up front, so memory use stays flat no matter
    how large `n_sims` is.

    Args:
        n_sims: Number of galaxies to generate parameters for.
        n_stars: Mean number of stars to sample per galaxy
            (Poisson-distributed).
        rng: Random number generator to draw samples from.

    Yields:
        Tuples of (params, num_stars), one per galaxy.
    """
    for _ in range(n_sims):
        params = rng.uniform(PRIOR_MIN, PRIOR_MAX)
        num_stars = int(rng.poisson(n_stars))
        yield params, num_stars


def main() -> None:
    """ Sample the 6D stellar kinematics of dwarf galaxies. """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is None:
        # record a concrete seed so an unseeded run can still be
        # reproduced from the saved config
        args.seed = int(np.random.SeedSequence().entropy)
    print(f'Using seed: {args.seed}')

    rng = np.random.default_rng(args.seed)
    param_stream = iter_params(args.n_sims, args.n_stars, rng)

    theta_buffer: list[np.ndarray] = []
    posvel_buffer: list[np.ndarray] = []
    n_success = 0

    # append mode: find the next available file index to avoid overwriting
    if not args.append:
        file_idx = 0
        save_config(
            args, os.path.join(args.output_dir, 'config.0.json'))
    else:
        if args.output_dir is None or not os.path.exists(args.output_dir):
            raise ValueError(
                f'Output directory {args.output_dir} does not exist, '
                'cannot append.')
        existing_files = glob(os.path.join(args.output_dir, 'data.*.h5'))
        existing_indices = [
            int(f.split('.')[1]) for f in existing_files
            if f.split('.')[1].isdigit()
        ]
        file_idx = max(existing_indices, default=-1) + 1

        existing_configs = glob(os.path.join(args.output_dir, 'config.*.json'))
        existing_config_indices = [
            int(f.split('.')[1]) for f in existing_configs
            if f.split('.')[1].isdigit()
        ]
        config_idx = max(existing_config_indices, default=-1) + 1
        save_config(
            args, os.path.join(args.output_dir, f'config.{config_idx:d}.json'))


    def _flush() -> None:
        nonlocal file_idx
        if not theta_buffer:
            return
        file_path = os.path.join(
            args.output_dir, f'data.{file_idx:d}.h5')
        save_simdata(theta_buffer, posvel_buffer, file_path)
        file_idx += 1
        theta_buffer.clear()
        posvel_buffer.clear()

    # cap the number of in-flight tasks so memory use stays bounded
    # no matter how large n_sims is
    max_pending = args.max_pending or args.n_workers * 4

    with ProcessPoolExecutor(
        max_workers=args.n_workers,
        initializer=_init_worker,
        initargs=(args.sample_threads,),
    ) as pool:
        pending = {
            pool.submit(simulate_one, p, n)
            for p, n in itertools.islice(param_stream, max_pending)
        }
        with tqdm(total=args.n_sims, desc='Simulating') as pbar:
            while pending:
                done, pending = wait(
                    pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    pbar.update(1)
                    params, pv = fut.result()
                    if pv is not None:
                        theta_buffer.append(params)
                        posvel_buffer.append(pv)
                        n_success += 1
                        if len(theta_buffer) >= args.galaxies_per_file:
                            _flush()
                    next_item = next(param_stream, None)
                    if next_item is not None:
                        p, n = next_item
                        pending.add(pool.submit(simulate_one, p, n))

    _flush()

    print(f'Successful simulations: {n_success} / {args.n_sims} '
          f'({n_success / args.n_sims * 100:.1f}%)')
    print(f'Saved {file_idx} file(s) to {args.output_dir}')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Time taken: {t2 - t1:.2f} seconds')
