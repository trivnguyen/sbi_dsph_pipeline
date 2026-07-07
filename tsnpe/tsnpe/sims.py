"""Simulate the 6D Cartesian stellar kinematics of dwarf galaxies (agama).

Same physics/preprocessing as npe/simulate_8params_process.py, plus a
batch runner + HDF5 writer for simulate_round.py. Output is raw Cartesian
pos/vel; sky-plane projection happens at train time as a pre_transform,
not here (see tsnpe/proposal.py's docstring).
"""

import os

# Must be set before `import agama`, so every worker process defaults to
# single-threaded OpenMP regardless of fork timing.
os.environ.setdefault('OMP_NUM_THREADS', '1')

from concurrent.futures import ProcessPoolExecutor, as_completed

import agama
import astropy.units as u
import h5py
import numpy as np
from tqdm import tqdm

agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)
agama.setNumThreads(1)


def _init_worker(sample_threads: int) -> None:
    """Pool initializer: run once in each worker process at startup."""
    agama.setNumThreads(sample_threads)


def simulator(num_stars: int, params: np.ndarray) -> np.ndarray | None:
    """Sample 6D positions and velocities for stars in a dwarf galaxy.

    Args:
        num_stars: Number of stars to draw from the galaxy model.
        params: Array of 8 physical parameters (alpha, beta, gamma,
            log_rdm, log_rhos, beta0, log_ra, log_rstar) - the same
            ordering as tsnpe.prior.ALL_PARAM_NAMES.

    Returns:
        Array of shape (num_stars, 6) with columns (x, y, z, vx, vy, vz),
        or None if the model could not be constructed.
    """
    alpha, beta, gamma, log_rdm, log_rhos, beta0, log_ra, log_rstar = params
    r_dm = 10 ** log_rdm
    r_star = 10 ** log_rstar
    r_a = 10 ** log_ra
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
    posvel: np.ndarray, params: np.ndarray,
    vrange: tuple[float, float] = (0., 1000.),
    vdisp_range: tuple[float, float] = (1e-10, 1e10),
    r_rstar_range: tuple[float, float] = (0., 10.),
    r_kpc_range: tuple[float, float] = (0., 100.),
    min_frac_kept: float = 0.5,
) -> np.ndarray | None:
    """Filter a simulated galaxy against basic acceptance criteria.

    Args:
        posvel: (num_stars, 6) array as returned by `simulator`.
        params: The same 8 physical parameters passed to `simulator`.
        vrange: Accepted 3D speed range, km/s.
        vdisp_range: Accepted 3D velocity-dispersion range, km/s.
        r_rstar_range: Accepted 3D radius range, in units of r_star.
        r_kpc_range: Accepted 3D radius range, kpc. The tighter of this and
            `r_rstar_range` (converted to kpc) applies.
        min_frac_kept: Minimum fraction of stars that must pass the
            velocity/radius mask for the galaxy to be accepted.

    Returns:
        Filtered (n_kept, 6) posvel array, or None if the galaxy is
        rejected (velocity dispersion out of range, or too few stars pass).
    """
    log_rstar = params[-1]
    r_star = 10 ** log_rstar

    num_stars = posvel.shape[0]
    rad3d = np.linalg.norm(posvel[:, :3], axis=1)
    vel3d = np.linalg.norm(posvel[:, 3:6], axis=1)
    veldisp3d = np.std(vel3d)

    if not (vdisp_range[0] < veldisp3d < vdisp_range[1]):
        return None

    min_radius = max(r_rstar_range[0] * r_star, r_kpc_range[0])
    max_radius = min(r_rstar_range[1] * r_star, r_kpc_range[1])

    mask = (
        (vel3d > vrange[0]) & (vel3d < vrange[1])
        & (rad3d > min_radius) & (rad3d < max_radius)
    )
    if np.sum(mask) < int(num_stars * min_frac_kept):
        return None
    return posvel[mask]


def simulate_one(
    params: np.ndarray, num_stars: int, preprocess_kwargs: dict | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Simulate and preprocess a single galaxy.

    Runs in a worker process. Defined at module level so it can be pickled
    and sent to worker processes by `ProcessPoolExecutor`. No locking
    needed - each process has isolated agama state.

    Args:
        params: Array of 8 physical parameters.
        num_stars: Number of stars to sample for this galaxy.
        preprocess_kwargs: Extra kwargs forwarded to `preprocess`.

    Returns:
        (params, posvel) if accepted, or (None, None) if rejected at
        either simulation or preprocessing.
    """
    pv = simulator(num_stars, params)
    if pv is None:
        return None, None
    pv = preprocess(pv, params, **(preprocess_kwargs or {}))
    if pv is None:
        return None, None
    return params, pv


def run_simulation_batch(
    params: np.ndarray,
    num_stars: list[int],
    n_jobs: int = 0,
    use_multiprocessing: bool = True,
    sample_threads: int = 1,
    preprocess_kwargs: dict | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Simulate a batch of galaxies, keeping only the successful draws.

    Args:
        params: (N, 8) array of physical parameters, one row per galaxy
            (see `simulator` for column order).
        num_stars: Length-N sequence of star counts to draw per galaxy.
        n_jobs: Worker processes to use (0 -> os.cpu_count()). Ignored if
            `use_multiprocessing` is False.
        use_multiprocessing: If False, simulate serially in this process
            (useful for debugging).
        sample_threads: OpenMP threads each worker process may use
            internally for agama calls.
        preprocess_kwargs: Extra kwargs forwarded to `preprocess`.

    Returns:
        Tuple of (theta, posvel_list): `theta` is the (n_success, 8) subset
        of `params` that were accepted, and `posvel_list` is the matching
        list of (n_i, 6) position/velocity arrays.
    """
    params = np.asarray(params)
    theta_list, posvel_list = [], []

    if not use_multiprocessing:
        for p, n in tqdm(
            zip(params, num_stars), total=len(params), desc='Simulating',
        ):
            p_out, pv = simulate_one(p, int(n), preprocess_kwargs)
            if pv is not None:
                theta_list.append(p_out)
                posvel_list.append(pv)
        return np.array(theta_list), posvel_list

    n_workers = n_jobs or os.cpu_count()
    with ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker,
        initargs=(sample_threads,),
    ) as pool:
        futures = [
            pool.submit(simulate_one, p, int(n), preprocess_kwargs)
            for p, n in zip(params, num_stars)
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc='Simulating'):
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
        theta: (n_galaxies, len(param_names)) physical-unit parameters.
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
