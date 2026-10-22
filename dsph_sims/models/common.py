"""agama setup and the pieces every family shares.

Importing this module configures agama's units and thread count, so the
family modules import it before they touch agama; the batch runner's
worker initializer re-applies the thread count per process.
"""

import os

# Must be set before `import agama`, so every worker process defaults to
# single-threaded OpenMP regardless of fork timing.
os.environ.setdefault('OMP_NUM_THREADS', '1')

import agama  # noqa: E402
import astropy.units as u  # noqa: E402
import numpy as np  # noqa: E402

agama.setUnits(mass=1 * u.Msun, length=1 * u.kpc, velocity=1 * u.km / u.s)
agama.setNumThreads(1)


def init_worker(sample_threads: int) -> None:
    """Pool initializer: OpenMP threads each worker may use for agama."""
    agama.setNumThreads(sample_threads)


def preprocess(
    posvel: np.ndarray,
    r_star: float,
    vdisp_range: tuple[float, float] = (1e-10, 1e10),
    vdisp_method: str = 'speed_std',
    vrange: tuple[float, float] = (0., 200.),
    r_rstar_range: tuple[float, float] = (0., 10.),
    r_kpc_range: tuple[float, float] = (0., 100.),
    min_frac_kept: float = 0.5,
) -> 'np.ndarray | None':
    """Acceptance cuts shared by every family, as the scripts applied them.

    Args:
        posvel: (num_stars, 6) Cartesian pos/vel from `simulate`.
        r_star: Tracer scale radius in kpc (each family knows its own
            units and converts before calling).
        vdisp_range: Accepted 3D velocity dispersion, km/s. A galaxy
            outside it is rejected outright.
        vdisp_method: 'speed_std' (std of |v|) or 'component_rss'
            (sqrt of the summed per-component variances; priorC).
        vrange: Accepted 3D speed per star, km/s.
        r_rstar_range: Accepted 3D radius per star, in r_star units.
        r_kpc_range: Accepted 3D radius per star, kpc. The tighter of
            this and `r_rstar_range` applies.
        min_frac_kept: Reject the galaxy if fewer than this fraction of
            its stars pass the per-star cuts.

    Returns:
        The stars that pass, or None if the galaxy is rejected.

    Raises:
        ValueError: If `vdisp_method` is unknown.
    """
    num_stars = posvel.shape[0]
    rad3d = np.linalg.norm(posvel[:, :3], axis=1)
    vel3d = np.linalg.norm(posvel[:, 3:6], axis=1)
    if vdisp_method == 'speed_std':
        veldisp3d = np.std(vel3d)
    elif vdisp_method == 'component_rss':
        veldisp3d = np.sqrt(np.sum(np.var(posvel[:, 3:6], axis=0)))
    else:
        raise ValueError(
            f'vdisp_method={vdisp_method!r} not recognized; must be '
            "'speed_std' or 'component_rss'")

    # Reject-if rather than accept-if, exactly as the scripts wrote it,
    # so a NaN dispersion (a zero-star draw) passes here as it did there.
    if (veldisp3d < vdisp_range[0]) or (veldisp3d > vdisp_range[1]):
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


def half_light_radius(
    density: agama.Density, r_max: float, num_grid: int = 200,
) -> float:
    """Projected half-light radius of a spherical unit-mass tracer.

    Args:
        density: agama `Density` of the tracer.
        r_max: Outer radius of the integration grid, kpc.
        num_grid: Logarithmically spaced grid points.

    Returns:
        Radius enclosing half the projected light, kpc.
    """
    grid = np.logspace(np.log10(r_max) - 6, np.log10(r_max), num_grid)
    sigma = density.projectedDensity(
        np.column_stack([grid, np.zeros_like(grid)]))
    # cumulative projected mass, 2 pi int Sigma R dR by the trapezoid rule
    integrand = 2 * np.pi * sigma * grid
    cumulative = np.concatenate([[0], np.cumsum(
        0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    return float(np.interp(0.5 * cumulative[-1], cumulative, grid))
